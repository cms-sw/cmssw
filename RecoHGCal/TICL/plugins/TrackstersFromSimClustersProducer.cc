// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 02/2021

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

#include "TrackstersPCA.h"
#include <vector>
#include <iterator>
#include <algorithm>
using namespace ticl;

namespace {
  Trackster::ParticleType tracksterParticleTypeFromPdgId(int pdgId, int charge) {
    if (pdgId == 111) {
      return Trackster::ParticleType::neutral_pion;
    } else {
      pdgId = std::abs(pdgId);
      if (pdgId == 22) {
        return Trackster::ParticleType::photon;
      } else if (pdgId == 11) {
        return Trackster::ParticleType::electron;
      } else if (pdgId == 13) {
        return Trackster::ParticleType::muon;
      } else {
        bool isHadron = (pdgId > 100 and pdgId < 900) or (pdgId > 1000 and pdgId < 9000);
        if (isHadron) {
          if (charge != 0) {
            return Trackster::ParticleType::charged_hadron;
          } else {
            return Trackster::ParticleType::neutral_hadron;
          }
        } else {
          return Trackster::ParticleType::unknown;
        }
      }
    }
  }
}  // namespace

class TrackstersFromSimClustersProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackstersFromSimClustersProducer(const edm::ParameterSet&);
  ~TrackstersFromSimClustersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::string detector_;
  const bool doNose_ = false;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;

  edm::EDGetTokenT<std::vector<SimCluster>> simclusters_token_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloparticles_token_;

  edm::InputTag associatorLayerClusterSimCluster_;
  edm::EDGetTokenT<hgcal::SimToRecoCollectionWithSimClusters> associatorMapSimClusterToReco_token_;
  edm::InputTag associatorLayerClusterCaloParticle_;
  edm::EDGetTokenT<hgcal::SimToRecoCollection> associatorMapCaloParticleToReco_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  hgcal::RecHitTools rhtools_;
  const double fractionCut_;
};
DEFINE_FWK_MODULE(TrackstersFromSimClustersProducer);

TrackstersFromSimClustersProducer::TrackstersFromSimClustersProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      filtered_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("filtered_mask"))),
      simclusters_token_(consumes<std::vector<SimCluster>>(ps.getParameter<edm::InputTag>("simclusters"))),
      caloparticles_token_(consumes<std::vector<CaloParticle>>(ps.getParameter<edm::InputTag>("caloparticles"))),
      associatorLayerClusterSimCluster_(ps.getUntrackedParameter<edm::InputTag>("layerClusterSimClusterAssociator")),
      associatorMapSimClusterToReco_token_(
          consumes<hgcal::SimToRecoCollectionWithSimClusters>(associatorLayerClusterSimCluster_)),
      associatorLayerClusterCaloParticle_(
          ps.getUntrackedParameter<edm::InputTag>("layerClusterCaloParticleAssociator")),
      associatorMapCaloParticleToReco_token_(consumes<hgcal::SimToRecoCollection>(associatorLayerClusterCaloParticle_)),
      geom_token_(esConsumes()),
      fractionCut_(ps.getParameter<double>("fractionCut")) {
  produces<std::vector<Trackster>>();
  produces<std::vector<float>>();
}

void TrackstersFromSimClustersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClustersSimTracksters", "ticlSimTracksters"));
  desc.add<edm::InputTag>("simclusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("caloparticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.addUntracked<edm::InputTag>("layerClusterSimClusterAssociator",
                                   edm::InputTag("layerClusterSimClusterAssociationProducer"));
  desc.addUntracked<edm::InputTag>("layerClusterCaloParticleAssociator",
                                   edm::InputTag("layerClusterCaloParticleAssociationProducer"));
  desc.add<double>("fractionCut", 0.);

  descriptions.add("trackstersFromSimClustersProducer", desc);
}

void TrackstersFromSimClustersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<Trackster>>();
  auto output_mask = std::make_unique<std::vector<float>>();
  const auto& layerClusters = evt.get(clusters_token_);
  const auto& layerClustersTimes = evt.get(clustersTime_token_);
  const auto& inputClusterMask = evt.get(filtered_layerclusters_mask_token_);
  output_mask->resize(layerClusters.size(), 1.f);

  const auto& simclusters = evt.get(simclusters_token_);
  const auto& caloparticles = evt.get(caloparticles_token_);

  const auto& simClustersToRecoColl = evt.get(associatorMapSimClusterToReco_token_);
  const auto& caloParticlesToRecoColl = evt.get(associatorMapCaloParticleToReco_token_);

  const auto& geom = es.getData(geom_token_);
  rhtools_.setGeometry(geom);
  auto num_simclusters = simclusters.size();
  result->reserve(num_simclusters);

  for (const auto& [key, values] : caloParticlesToRecoColl) {
    auto const& cp = *(key);
    auto cpIndex = &cp - &caloparticles[0];
    if (cp.g4Tracks()[0].crossedBoundary()) {
      if (values.empty())
        continue;
      Trackster tmpTrackster;
      tmpTrackster.zeroProbabilities();
      tmpTrackster.vertices().reserve(values.size());
      tmpTrackster.vertex_multiplicity().reserve(values.size());
      for (auto const& [lc, energyScorePair] : values) {
        if (inputClusterMask[lc.index()] > 0) {
          double fraction = energyScorePair.first / lc->energy();
          if (fraction < fractionCut_)
            continue;
          tmpTrackster.vertices().push_back(lc.index());
          (*output_mask)[lc.index()] -= fraction;
          tmpTrackster.vertex_multiplicity().push_back(1. / fraction);
        }
      }
      tmpTrackster.setIdProbability(tracksterParticleTypeFromPdgId(cp.pdgId(), cp.charge()), 1.f);
      float energyAtBoundary = cp.g4Tracks()[0].getMomentumAtBoundary().energy();
      tmpTrackster.setRegressedEnergy(energyAtBoundary);
      tmpTrackster.setSeed(key.id(), cpIndex);
      result->emplace_back(tmpTrackster);
    } else {
      for (const auto& scRef : cp.simClusters()) {
        const auto& it = simClustersToRecoColl.find(scRef);
        if (it == simClustersToRecoColl.end())
          continue;
        const auto& lcVec = it->val;
        if (lcVec.empty())
          continue;
        auto const& sc = *(scRef);
        auto simClusterIndex = &sc - &simclusters[0];
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
            (*output_mask)[lc.index()] -= fraction;
            tmpTrackster.vertex_multiplicity().push_back(1. / fraction);
          }
        }
        tmpTrackster.setIdProbability(tracksterParticleTypeFromPdgId(sc.pdgId(), sc.charge()), 1.f);
        float energyAtBoundary = sc.g4Tracks()[0].getMomentumAtBoundary().energy();
        tmpTrackster.setRegressedEnergy(energyAtBoundary);
        tmpTrackster.setSeed(scRef.id(), simClusterIndex);
        result->emplace_back(tmpTrackster);
      }
    }
  }

  ticl::assignPCAtoTracksters(
      *result, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE(doNose_)).z());
  result->shrink_to_fit();

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
}
