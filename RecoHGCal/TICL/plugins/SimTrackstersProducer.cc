// Author: Felice Pantaleo, Leonardo Cristella - felice.pantaleo@cern.ch, leonardo.cristella@cern.ch
// Date: 09/2021

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
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "RecoHGCal/TICL/interface/commons.h"

#include "TrackstersPCA.h"
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>

using namespace ticl;

class SimTrackstersProducer : public edm::stream::EDProducer<> {
public:
  explicit SimTrackstersProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;
  void addTrackster(const int& index,
                    const std::vector<std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>>& lcVec,
                    const std::vector<float>& inputClusterMask,
                    const float& fractionCut_,
                    const float& energy,
                    const int& pdgId,
                    const int& charge,
                    const edm::ProductID& seed,
                    const Trackster::IterationIndex iter,
                    std::vector<float>& output_mask,
                    std::vector<Trackster>& result);

private:
  std::string detector_;
  const bool doNose_ = false;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;

  const edm::EDGetTokenT<std::vector<SimCluster>> simclusters_token_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloparticles_token_;

  const edm::EDGetTokenT<hgcal::SimToRecoCollectionWithSimClusters> associatorMapSimClusterToReco_token_;
  const edm::EDGetTokenT<hgcal::SimToRecoCollection> associatorMapCaloParticleToReco_token_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  hgcal::RecHitTools rhtools_;
  const double fractionCut_;
};
DEFINE_FWK_MODULE(SimTrackstersProducer);

SimTrackstersProducer::SimTrackstersProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      clusters_token_(consumes(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(consumes(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      filtered_layerclusters_mask_token_(consumes(ps.getParameter<edm::InputTag>("filtered_mask"))),
      simclusters_token_(consumes(ps.getParameter<edm::InputTag>("simclusters"))),
      caloparticles_token_(consumes(ps.getParameter<edm::InputTag>("caloparticles"))),
      associatorMapSimClusterToReco_token_(
          consumes(ps.getParameter<edm::InputTag>("layerClusterSimClusterAssociator"))),
      associatorMapCaloParticleToReco_token_(
          consumes(ps.getParameter<edm::InputTag>("layerClusterCaloParticleAssociator"))),
      geom_token_(esConsumes()),
      fractionCut_(ps.getParameter<double>("fractionCut")) {
  produces<TracksterCollection>();
  produces<std::vector<float>>();
  produces<TracksterCollection>("fromCPs");
  produces<std::vector<float>>("fromCPs");
  produces<std::map<uint, std::vector<uint>>>();
}

void SimTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClustersSimTracksters", "ticlSimTracksters"));
  desc.add<edm::InputTag>("simclusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("caloparticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("layerClusterSimClusterAssociator",
                          edm::InputTag("layerClusterSimClusterAssociationProducer"));
  desc.add<edm::InputTag>("layerClusterCaloParticleAssociator",
                          edm::InputTag("layerClusterCaloParticleAssociationProducer"));
  desc.add<double>("fractionCut", 0.);

  descriptions.addWithDefaultLabel(desc);
}

void SimTrackstersProducer::addTrackster(
    const int& index,
    const std::vector<std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>>& lcVec,
    const std::vector<float>& inputClusterMask,
    const float& fractionCut_,
    const float& energy,
    const int& pdgId,
    const int& charge,
    const edm::ProductID& seed,
    const Trackster::IterationIndex iter,
    std::vector<float>& output_mask,
    std::vector<Trackster>& result) {
  if (lcVec.empty())
    return;

  Trackster tmpTrackster;
  tmpTrackster.zeroProbabilities();
  tmpTrackster.vertices().reserve(lcVec.size());
  tmpTrackster.vertex_multiplicity().reserve(lcVec.size());
  for (auto const& [lc, energyScorePair] : lcVec) {
    if (inputClusterMask[lc.index()] > 0) {
      double fraction = energyScorePair.first / lc->energy();
      if (fraction < fractionCut_)
        continue;
      tmpTrackster.vertices().push_back(lc.index());
      output_mask[lc.index()] -= fraction;
      tmpTrackster.vertex_multiplicity().push_back(1. / fraction);
    }
  }

  tmpTrackster.setIdProbability(tracksterParticleTypeFromPdgId(pdgId, charge), 1.f);
  tmpTrackster.setRegressedEnergy(energy);
  tmpTrackster.setIteration(iter);
  tmpTrackster.setSeed(seed, index);
  result.emplace_back(tmpTrackster);
}

void SimTrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<TracksterCollection>();
  auto output_mask = std::make_unique<std::vector<float>>();
  auto result_fromCP = std::make_unique<TracksterCollection>();
  auto output_mask_fromCP = std::make_unique<std::vector<float>>();
  auto cpToSc_SimTrackstersMap = std::make_unique<std::map<uint, std::vector<uint>>>();
  const auto& layerClusters = evt.get(clusters_token_);
  const auto& layerClustersTimes = evt.get(clustersTime_token_);
  const auto& inputClusterMask = evt.get(filtered_layerclusters_mask_token_);
  output_mask->resize(layerClusters.size(), 1.f);
  output_mask_fromCP->resize(layerClusters.size(), 1.f);

  const auto& simclusters = evt.get(simclusters_token_);
  const auto& caloparticles = evt.get(caloparticles_token_);

  const auto& simClustersToRecoColl = evt.get(associatorMapSimClusterToReco_token_);
  const auto& caloParticlesToRecoColl = evt.get(associatorMapCaloParticleToReco_token_);

  const auto& geom = es.getData(geom_token_);
  rhtools_.setGeometry(geom);
  const auto num_simclusters = simclusters.size();
  result->reserve(num_simclusters);  // Conservative size, will call shrink_to_fit later
  const auto num_caloparticles = caloparticles.size();
  result_fromCP->reserve(num_caloparticles);

  for (const auto& [key, lcVec] : caloParticlesToRecoColl) {
    auto const& cp = *(key);
    auto cpIndex = &cp - &caloparticles[0];

    auto regr_energy = cp.energy();
    std::vector<uint> scSimTracksterIdx;
    scSimTracksterIdx.reserve(cp.simClusters().size());

    // Create a Trackster from the object entering HGCal
    if (cp.g4Tracks()[0].crossedBoundary()) {
      regr_energy = cp.g4Tracks()[0].getMomentumAtBoundary().energy();

      addTrackster(cpIndex,
                   lcVec,
                   inputClusterMask,
                   fractionCut_,
                   regr_energy,
                   cp.pdgId(),
                   cp.charge(),
                   key.id(),
                   ticl::Trackster::SIM,
                   *output_mask,
                   *result);
    } else {
      for (const auto& scRef : cp.simClusters()) {
        const auto& it = simClustersToRecoColl.find(scRef);
        if (it == simClustersToRecoColl.end())
          continue;
        const auto& lcVec = it->val;
        auto const& sc = *(scRef);
        auto const scIndex = &sc - &simclusters[0];

        addTrackster(scIndex,
                     lcVec,
                     inputClusterMask,
                     fractionCut_,
                     sc.g4Tracks()[0].getMomentumAtBoundary().energy(),
                     sc.pdgId(),
                     sc.charge(),
                     scRef.id(),
                     ticl::Trackster::SIM,
                     *output_mask,
                     *result);

        if (result->empty())
          continue;
        const auto index = result->size() - 1;
        if (std::find(scSimTracksterIdx.begin(), scSimTracksterIdx.end(), index) == scSimTracksterIdx.end()) {
          scSimTracksterIdx.emplace_back(index);
        }
      }
      scSimTracksterIdx.shrink_to_fit();
    }

    // Create a Trackster from any CP
    addTrackster(cpIndex,
                 lcVec,
                 inputClusterMask,
                 fractionCut_,
                 regr_energy,
                 cp.pdgId(),
                 cp.charge(),
                 key.id(),
                 ticl::Trackster::SIM_CP,
                 *output_mask_fromCP,
                 *result_fromCP);

    if (result_fromCP->empty())
      continue;
    const auto index = result_fromCP->size() - 1;
    if (cpToSc_SimTrackstersMap->find(index) == cpToSc_SimTrackstersMap->end()) {
      (*cpToSc_SimTrackstersMap)[index] = scSimTracksterIdx;
    }
  }

  ticl::assignPCAtoTracksters(
      *result, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE(doNose_)).z());
  result->shrink_to_fit();
  ticl::assignPCAtoTracksters(
      *result_fromCP, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE(doNose_)).z());
  result_fromCP->shrink_to_fit();

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
  evt.put(std::move(result_fromCP), "fromCPs");
  evt.put(std::move(output_mask_fromCP), "fromCPs");
  evt.put(std::move(cpToSc_SimTrackstersMap));
}
