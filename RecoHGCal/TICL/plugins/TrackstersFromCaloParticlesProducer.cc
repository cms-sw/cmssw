// Author: Leonardo Cristella - leonardo.cristella@cern.ch
// Date: 08/2021

// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "RecoHGCal/TICL/interface/commons.h"

#include "TrackstersPCA.h"
#include <vector>
#include <iterator>
#include <algorithm>

using namespace ticl;

class TrackstersFromCaloParticlesProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackstersFromCaloParticlesProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::string detector_;
  const bool doNose_ = false;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;

  const edm::EDGetTokenT<std::vector<CaloParticle>> caloparticles_token_;

  const edm::EDGetTokenT<hgcal::SimToRecoCollection> associatorMapCaloParticleToReco_token_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  hgcal::RecHitTools rhtools_;
  const double fractionCut_;
};
DEFINE_FWK_MODULE(TrackstersFromCaloParticlesProducer);

TrackstersFromCaloParticlesProducer::TrackstersFromCaloParticlesProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      clusters_token_(consumes(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(consumes(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      filtered_layerclusters_mask_token_(consumes(ps.getParameter<edm::InputTag>("filtered_mask"))),
      caloparticles_token_(consumes(ps.getParameter<edm::InputTag>("caloparticles"))),
      associatorMapCaloParticleToReco_token_(consumes(ps.getParameter<edm::InputTag>("associator"))),
      geom_token_(esConsumes()),
      fractionCut_(ps.getParameter<double>("fractionCut")) {
  produces<std::vector<Trackster>>();
  produces<std::vector<float>>();
}

void TrackstersFromCaloParticlesProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClustersSimTracksters", "ticlSimTracksters"));
  desc.add<edm::InputTag>("caloparticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("associator", edm::InputTag("layerClusterCaloParticleAssociationProducer"));
  desc.add<double>("fractionCut", 0.);

  descriptions.addWithDefaultLabel(desc);
}

void TrackstersFromCaloParticlesProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<Trackster>>();
  auto output_mask = std::make_unique<std::vector<float>>();
  const auto& layerClusters = evt.get(clusters_token_);
  const auto& layerClustersTimes = evt.get(clustersTime_token_);
  const auto& inputClusterMask = evt.get(filtered_layerclusters_mask_token_);
  output_mask->resize(layerClusters.size(), 1.f);

  const auto& caloparticles = evt.get(caloparticles_token_);

  const auto& caloParticlesToRecoColl = evt.get(associatorMapCaloParticleToReco_token_);

  const auto& geom = es.getData(geom_token_);
  rhtools_.setGeometry(geom);
  result->reserve(caloparticles.size());

  for (const auto& [key, values] : caloParticlesToRecoColl) {
    auto const& cp = *(key);
    auto cpIndex = &cp - &caloparticles[0];

    addTrackster(cpIndex,
                 values,
                 inputClusterMask,
                 fractionCut_,
                 cp.g4Tracks()[0].getMomentumAtBoundary().energy(),
                 cp.pdgId(),
                 cp.charge(),
                 key.id(),
                 *output_mask,
                 *result);
  }

  ticl::assignPCAtoTracksters(
      *result, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE(doNose_)).z());
  result->shrink_to_fit();

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
}
