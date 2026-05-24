/**
* Produce tracksters (in reco ticl::Trackster dataformat) from simulation truth objects.
* Multiple simTrackster collections can be built, from different simulation truth collections

* Author: Felice Pantaleo, Leonardo Cristella - felice.pantaleo@cern.ch, leonardo.cristella@cern.ch
* Date: 09/2021
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterCollection.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/UniqueSimTrackId.h"

#include "DataFormats/HGCalReco/interface/Common.h"

#include <CLHEP/Units/SystemOfUnits.h>

#include "TrackstersPCA.h"

#include <numeric>
#include <vector>
#include <algorithm>

using namespace ticl;

namespace {
  /** Association map from SimCluster/CaloParticle to layer cluster (with (score, sharedEnergy) pair) */
  template <typename SimCaloObject_t>
  using SimToRecoLCAssociation = edm::AssociationMap<edm::OneToManyWithQualityGeneric<std::vector<SimCaloObject_t>,
                                                                                      reco::CaloClusterCollection,
                                                                                      std::pair<float, float>>>;

  /** Defines what is stored in a SimTrackster boundaryTime field. */
  enum class BoundaryTimeMode {
    none,           ///< nothing is stored
    simVertexTime,  ///< time of the SimVertex that created the GenParticle
    boundaryTime    ///< time of the SimTrack as it crossed calorimeter boundary
  };
  BoundaryTimeMode stringBoundaryTimeModeToEnum(std::string s) {
    if (s == "none")
      return BoundaryTimeMode::none;
    if (s == "simVertexTime")
      return BoundaryTimeMode::simVertexTime;
    if (s == "boundaryTime")
      return BoundaryTimeMode::boundaryTime;
    throw cms::Exception("Configuration") << "Unkown BoundaryTimeMode";
  }

  /** Configuration for producing one SimTrackster collection */
  struct SimTsConfig {
    /// Simulation truth calorimeter object type (here hardcoded to SimCluster but could be updated/templated)
    using SimCaloObject_t = SimCluster;
    using SimCaloObjectCollection_t = SimClusterCollection;

    SimTsConfig(edm::ParameterSet const& ps,
                edm::ConsumesCollector consumesCollector,
                edm::ProducesCollector prodCollector)
        : simclusters_token(consumesCollector.consumes<SimCaloObjectCollection_t>(
              ps.getParameter<edm::InputTag>("simClusterCollection"))),

          associatorMapSimClusterToReco_token(consumesCollector.consumes<SimToRecoLCAssociation<SimCaloObject_t>>(
              ps.getParameter<edm::InputTag>("simClusterToLayerClusterAssociationMap"))),
          simClusterToCaloParticleSC_map_token(
              consumesCollector.consumes<SimClusterRefVector>(ps.getParameter<edm::InputTag>("simClusterCollection"))),

          simTrackster_token(
              prodCollector.produces<TracksterCollection>(ps.getParameter<std::string>("outputProductLabel"))),
          outputMask_token(
              prodCollector.produces<std::vector<float>>(ps.getParameter<std::string>("outputProductLabel"))),
          simTracksterToSimCluster_map_token(prodCollector.produces<edm::RefVector<std::vector<SimCaloObject_t>>>(
              ps.getParameter<std::string>("outputProductLabel"))),
          simTracksterToCaloParticle_map_token(
              prodCollector.produces<CaloParticleRefVector>(ps.getParameter<std::string>("outputProductLabel"))),

          simTracksterBoundaryTime(
              stringBoundaryTimeModeToEnum(ps.getParameter<std::string>("simTracksterBoundaryTime"))),
          tracksterIterationIndex(
              static_cast<ticl::Trackster::IterationIndex>(ps.getParameter<int>("tracksterIterationIndex"))) {}

    const edm::EDGetTokenT<SimCaloObjectCollection_t> simclusters_token;
    const edm::EDGetTokenT<SimToRecoLCAssociation<SimCaloObject_t>> associatorMapSimClusterToReco_token;
    /// Map from simclusters_token collection to CaloParticle collection (here as SimCluster dataformat but 1-1 mapping to CaloParticle)
    const edm::EDGetTokenT<SimClusterRefVector> simClusterToCaloParticleSC_map_token;

    const edm::EDPutTokenT<TracksterCollection> simTrackster_token;  ///< output collection
    /// output layer cluster mask after masking LCs from SimCluster
    const edm::EDPutTokenT<std::vector<float>> outputMask_token;
    // output map from SimTrackster to SimCluster it was made from (1-1 mapping except when empty simts are removed)
    const edm::EDPutTokenT<edm::RefVector<std::vector<SimCaloObject_t>>> simTracksterToSimCluster_map_token;
    /// output Map from SimTrackster to CaloParticle (for convenience, can be recomputed by chaining maps SimTs->SimCluster->CaloParticle)
    const edm::EDPutTokenT<CaloParticleRefVector> simTracksterToCaloParticle_map_token;

    const BoundaryTimeMode simTracksterBoundaryTime;                ///< configuration for setting boundary time
    const ticl::Trackster::IterationIndex tracksterIterationIndex;  ///< to be set in each output trackster
  };
};  // namespace

class SimTrackstersProducer : public edm::stream::EDProducer<> {
public:
  explicit SimTrackstersProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /** Holds input collection, convenience struct */
  struct InputHolder {
    std::vector<reco::CaloCluster> const& layerClusters;
    edm::ValueMap<std::pair<float, float>> const& layerClustersTimes;
    std::vector<float> const& filtered_layerclusters_mask;

    std::vector<TrackingParticle> const& trackingParticle;

    std::vector<reco::Track> const& recoTracks;

    // Maps between reco tracks, SimTrack (=Geant4 track) and TrackingParticle (=tracking truth object for tracks)
    reco::SimToRecoCollection const& TPtoRecoTrackMap;
    SimTrackToTPMap const& simTrackToTPMap;

    edm::Handle<CaloParticleCollection> caloParticles_h;
  };

  template <typename SimCaloObject_t>
  void produceOne(edm::Event& evt, const edm::EventSetup& es, InputHolder const& holder, SimTsConfig const& config);

  /** For HLT not all collections are always present, put empty collections in event in this case */
  void returnEmptyCollections(edm::Event& e, const int lcSize);

  const std::string detector_;
  const bool doNose_ = false;
  const bool doBarrel_ = false;
  const bool computeLocalTime_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  hgcal::RecHitTools rhtools_;
  const float fractionCut_;
  const float qualityCutTrack_;
  const edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;

  const edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken_;
  const StringCutObjectSelector<reco::Track> cutTk_;

  // Maps between reco tracks, SimTrack (=Geant4 track) and TrackingParticle (=tracking truth object for tracks)
  const edm::EDGetTokenT<reco::SimToRecoCollection> associatormapStRsToken_;
  const edm::EDGetTokenT<reco::RecoToSimCollection> associatormapRtSsToken_;
  const edm::EDGetTokenT<SimTrackToTPMap> associationSimTrackToTPToken_;

  const edm::EDGetTokenT<CaloParticleCollection> caloParticles_token_;
  std::vector<SimTsConfig> simTsConfigs_;
};

DEFINE_FWK_MODULE(SimTrackstersProducer);

SimTrackstersProducer::SimTrackstersProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      doBarrel_(detector_ == "Barrel"),
      computeLocalTime_(ps.getParameter<bool>("computeLocalTime")),
      clusters_token_(consumes(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(consumes(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      filtered_layerclusters_mask_token_(consumes(ps.getParameter<edm::InputTag>("filtered_mask"))),
      geom_token_(esConsumes()),
      fractionCut_(ps.getParameter<double>("fractionCut")),
      qualityCutTrack_(ps.getParameter<double>("qualityCutTrack")),
      trackingParticleToken_(
          consumes<std::vector<TrackingParticle>>(ps.getParameter<edm::InputTag>("trackingParticles"))),
      recoTracksToken_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("recoTracks"))),
      cutTk_(ps.getParameter<std::string>("cutTk")),
      associatormapStRsToken_(consumes(ps.getParameter<edm::InputTag>("tpToTrack"))),
      associationSimTrackToTPToken_(consumes(ps.getParameter<edm::InputTag>("simTrackToTPMap"))),
      caloParticles_token_(consumes<CaloParticleCollection>(ps.getParameter<edm::InputTag>("caloParticles"))) {
  auto const& configsParameterSets = ps.getParameter<std::vector<edm::ParameterSet>>("simClusterCollections");
  if (configsParameterSets.empty())
    edm::LogWarning("EmptyConfig")
        << "No config for input SimCluster collections. SimTrackstersProducer will not produce anything ";
  for (edm::ParameterSet const& configPs : configsParameterSets) {
    simTsConfigs_.emplace_back(configPs, consumesCollector(), producesCollector());
  }
}

void SimTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<bool>("computeLocalTime", "true");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClustersSimTracksters", "ticlSimTracksters"));
  desc.add<edm::InputTag>("recoTracks", edm::InputTag("generalTracks"));
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  desc.add<edm::InputTag>("tpToTrack", edm::InputTag("trackingParticleRecoTrackAsssociation"));

  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));

  desc.add<edm::InputTag>("simTrackToTPMap", edm::InputTag("simHitTPAssocProducer", "simTrackToTP"));
  desc.add<double>("fractionCut", 0.);
  desc.add<double>("qualityCutTrack", 0.75);

  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"))
      ->setComment("CaloParticle collection (only used to build 'direct' convenience map SimTrackster->CaloParticle)");

  // Settings for individual SimCluster collections
  edm::ParameterSetDescription simClusterDescValidator;
  simClusterDescValidator.add<edm::InputTag>("simClusterCollection")
      ->setComment("Input tag for the SimCluster collection to use");
  simClusterDescValidator.add<edm::InputTag>("simClusterToLayerClusterAssociationMap")
      ->setComment(
          "Association map simToReco mapping SimCluster (same collection as simClusterCollection) to layer cluster. "
          "Type : ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection> for SimCluster");
  simClusterDescValidator.add<std::string>("outputProductLabel")
      ->setComment("Product label for output SimTrackster collection");
  simClusterDescValidator
      .ifValue(edm::ParameterDescription<std::string>("simTracksterBoundaryTime", "none", true),
               edm::allowedValues<std::string>("none", "simVertexTime", "boundaryTime"))
      ->setComment(
          "Set what SimTrackster timeAtBoundary is set to (note that ts.time is always set to reco time from LCs). Can "
          "be 'none' (not set), 'simVertexTime' (SimVertex time of the SimTrack, aka CaloParticle.simTime(), only for "
          "CaloParticle) or 'boundaryTime' (time of SimTrack at boundary) ");
  simClusterDescValidator.add<int>("tracksterIterationIndex")
      ->setComment(
          "IterationIndex to use for Trackster output collection. See Trackster.h (ticl::Trackster::IterationIndex "
          "enum). 5=SIM (ie from SimCluster), 6=SIM_CP (ie from CaloParticle)");

  desc.addVPSet("simClusterCollections", simClusterDescValidator, {})  // default is set in python
      ->setComment("SimCluster collections to use for making SimTracksters");

  descriptions.addWithDefaultLabel(desc);
}

namespace {
  /** Create a trackster from the given layer cluster association elements. Also updates the layer cluster mask */
  template <typename SimCaloCluster_t>
  Trackster createTrackster(
      std::vector<std::pair<edm::Ref<std::vector<SimCaloCluster_t>>, std::pair<float, float>>> const& lcVec,
      const std::vector<float>& inputClusterMask,
      std::vector<float>& output_mask,
      float fractionCut_) {
    Trackster tmpTrackster;

    tmpTrackster.vertices().reserve(lcVec.size());
    tmpTrackster.vertex_multiplicity().reserve(lcVec.size());
    for (auto const& [lcRef, energyScorePair] : lcVec) {
      // lcRef is edm::Ref to CaloCluster, energyScorePair is pair<float, float>
      if (inputClusterMask[lcRef.index()] > 0) {
        float fraction = energyScorePair.first / lcRef->energy();
        if (fraction < fractionCut_)
          continue;
        tmpTrackster.vertices().push_back(lcRef.index());
        output_mask[lcRef.index()] -= fraction;
        tmpTrackster.vertex_multiplicity().push_back(1. / fraction);
      }
    }
    return tmpTrackster;
  }
};  // namespace

void SimTrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  const auto& layerClustersHandle = evt.getHandle(clusters_token_);
  const auto& layerClustersTimesHandle = evt.getHandle(clustersTime_token_);
  const auto& inputClusterMaskHandle = evt.getHandle(filtered_layerclusters_mask_token_);

  if (!layerClustersHandle.isValid() || !layerClustersTimesHandle.isValid() || !inputClusterMaskHandle.isValid()) {
    returnEmptyCollections(evt, 0);
    return;
  }

  edm::Handle<std::vector<TrackingParticle>> trackingParticles_h;
  evt.getByToken(trackingParticleToken_, trackingParticles_h);
  edm::Handle<std::vector<reco::Track>> recoTracks_h;
  evt.getByToken(recoTracksToken_, recoTracks_h);

  //TrackingParticle to reco track map
  const auto TPtoRecoTrackMapHandle = evt.getHandle(associatormapStRsToken_);

  if (!TPtoRecoTrackMapHandle.isValid()) {
    returnEmptyCollections(evt, layerClustersHandle->size());
    return;
  }

  const auto& geom = es.getData(geom_token_);
  rhtools_.setGeometry(geom);

  InputHolder inps = InputHolder{*layerClustersHandle,
                                 *layerClustersTimesHandle,
                                 *inputClusterMaskHandle,
                                 *trackingParticles_h,
                                 *recoTracks_h,
                                 evt.get(associatormapStRsToken_),
                                 evt.get(associationSimTrackToTPToken_),
                                 evt.getHandle(caloParticles_token_)};
  for (SimTsConfig const& conf : simTsConfigs_) {
    produceOne<SimCluster>(evt, es, inps, conf);
  }
}

template <typename SimCaloObject_t>
void SimTrackstersProducer::produceOne(edm::Event& evt,
                                       const edm::EventSetup& es,
                                       InputHolder const& holder,
                                       SimTsConfig const& config) {
  edm::Handle<std::vector<SimCaloObject_t>> simclusters_h;
  evt.getByToken(config.simclusters_token, simclusters_h);
  const auto& simclusters = *simclusters_h;

  auto const& simClusterToCaloParticleSC_map = evt.get(config.simClusterToCaloParticleSC_map_token);

  const SimToRecoLCAssociation<SimCaloObject_t>& simClustersToRecoColl =
      evt.get(config.associatorMapSimClusterToReco_token);  //*associatorMapSimClusterToReco_h;

  TracksterCollection simTracksters;  // output
  simTracksters.reserve(simclusters.size());

  std::vector<float> output_mask;
  output_mask.resize(holder.layerClusters.size(), 1.f);

  /// Output map : SimTrackster -> SimCluster used to build it
  edm::RefVector<std::vector<SimCaloObject_t>> simTracksterToSimObject_map(simclusters_h.id());
  simTracksterToSimObject_map.reserve(simclusters.size());
  CaloParticleRefVector simTracksterToCaloParticle_map(holder.caloParticles_h.id());
  simTracksterToCaloParticle_map.reserve(simclusters.size());

  for (const auto& [simClusterKey, lcVec] : simClustersToRecoColl) {
    // simClustersToRecoColl is ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection> (aka AssociationMap OneToManyWithQuality SimCluster -> many LayerCluster along with score,sharedE)
    // simClusterKey is simCluster ref, lcVec is std::vector<std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>>&
    SimCaloObject_t const& simCluster = *simClusterKey;

    if (lcVec.empty())
      continue;

    Trackster simTrackster = createTrackster(lcVec, holder.filtered_layerclusters_mask, output_mask, fractionCut_);
    if (simTrackster.vertices().empty())
      continue;  // The sim->reco associators expect non-empty tracksters

    if (simCluster.genParticles().empty()) {
      // Generic SimCluster case : encode SimTrack composition in id_probabilities of the SimTrackster (energy-weighted)
      std::array<float, Trackster::kParticleTypeLength> id_probs{};
      for (SimTrack const& simTrack : simCluster.g4Tracks())
        id_probs[static_cast<std::size_t>(tracksterParticleTypeFromPdgId(simTrack.type(), simTrack.charge()))] +=
            simTrack.momentum().E();
      float norm = 1. / std::accumulate(id_probs.begin(), id_probs.end(), 0.f);
      for (std::size_t i = 0; i < id_probs.size(); i++)
        id_probs[i] *= norm;
      simTrackster.setProbabilities(id_probs.data());
    } else {
      // In case SimCluster is linked to a genParticle (ie CaloParticle case): take info from there
      simTrackster.setIdProbability(
          tracksterParticleTypeFromPdgId(simCluster.genParticles()[0]->pdgId(), simCluster.genParticles()[0]->charge()),
          1.f);
    }

    if (simCluster.g4Tracks().at(0).crossedBoundary())
      simTrackster.setRegressedEnergy(simCluster.g4Tracks()[0].getMomentumAtBoundary().energy());
    else
      simTrackster.setRegressedEnergy(simCluster.energy());  // Taken from the momentum of first simTrack of SimCluster

    simTrackster.setIteration(config.tracksterIterationIndex);
    simTrackster.setSeed(simClusterKey.id(), simClusterKey.index());
    if (config.simTracksterBoundaryTime == BoundaryTimeMode::simVertexTime) {
      // SimCluster dataformat does not have SimTime so we rather take it from the CaloParticle
      // We use the 1-1 mapping between the SimCluster "CaloParticle" collection and the CaloParticle collection
      // TODO this could be simplified eg by adding simTime field in SimCluster ?
      simTrackster.setBoundaryTime(
          (*holder.caloParticles_h)[simClusterToCaloParticleSC_map[simClusterKey.index()].index()].simTime());
    } else if (config.simTracksterBoundaryTime == BoundaryTimeMode::boundaryTime)
      simTrackster.setBoundaryTime(simCluster.g4Tracks().at(0).getPositionAtBoundary().t() * CLHEP::s);

    simTracksters.emplace_back(std::move(simTrackster));
    simTracksterToSimObject_map.push_back(simClusterKey);
    simTracksterToCaloParticle_map.push_back(edm::Ref<CaloParticleCollection>(
        holder.caloParticles_h, simClusterToCaloParticleSC_map[simClusterKey.index()].index()));
  }
  // TODO: remove time computation from PCA calculation and
  //       store time from boundary position in simTracksters
  ticl::assignPCAtoTracksters(simTracksters,
                              holder.layerClusters,
                              holder.layerClustersTimes,
                              rhtools_.getPositionLayer(rhtools_.lastLayerEE(doNose_)).z(),
                              rhtools_,
                              computeLocalTime_,
                              true,
                              false,
                              doBarrel_);

  auto simTrackToRecoTrack = [&](UniqueSimTrackId simTkId) -> std::vector<int> {
    std::vector<int> trackIdx;
    auto ipos = holder.simTrackToTPMap.mapping.find(simTkId);
    if (ipos != holder.simTrackToTPMap.mapping.end()) {
      auto jpos = holder.TPtoRecoTrackMap.find((ipos->second));
      if (jpos != holder.TPtoRecoTrackMap.end()) {
        auto& associatedRecoTracks = jpos->val;
        if (!associatedRecoTracks.empty()) {
          // associated reco tracks are sorted by decreasing quality
          if (associatedRecoTracks[0].second > qualityCutTrack_) {
            trackIdx.push_back(&(*associatedRecoTracks[0].first) - &holder.recoTracks[0]);
          }
        }
      }
      const auto& tp = (*ipos->second);
      if (!tp.decayVertices().empty()) {
        const auto& iTV = tp.decayVertices()[0];
        for (auto iTP = iTV->daughterTracks_begin(); iTP != iTV->daughterTracks_end(); ++iTP) {
          auto kpos = holder.TPtoRecoTrackMap.find((*iTP));
          if (kpos != holder.TPtoRecoTrackMap.end()) {
            auto& associatedRecoTracks = kpos->val;
            if (!associatedRecoTracks.empty()) {
              // associated reco tracks are sorted by decreasing quality
              if (associatedRecoTracks[0].second > qualityCutTrack_) {
                trackIdx.push_back(&(*associatedRecoTracks[0].first) - &holder.recoTracks[0]);
              }
            }
          }
        }
      }
    }
    return trackIdx;
  };

  // Set the reco track id to SimTrackster
  for (unsigned int simTs_i = 0; simTs_i < simTracksters.size(); ++simTs_i) {
    Trackster& simTrackster = simTracksters[simTs_i];
    if (simTrackster.vertices().empty())
      continue;
    const auto& simTrack = simTracksterToSimObject_map[simTs_i]->g4Tracks()[0];
    UniqueSimTrackId simTkIds(simTrack.trackId(), simTrack.eventId());
    auto bestAssociatedRecoTracks = simTrackToRecoTrack(simTkIds);
    if (not bestAssociatedRecoTracks.empty()) {
      for (auto const trackIndex : bestAssociatedRecoTracks)
        simTrackster.addTrackIdx(trackIndex);
    }
  }

  evt.emplace(config.simTrackster_token, std::move(simTracksters));
  evt.emplace(config.outputMask_token, std::move(output_mask));
  evt.emplace(config.simTracksterToSimCluster_map_token, std::move(simTracksterToSimObject_map));
  evt.emplace(config.simTracksterToCaloParticle_map_token, std::move(simTracksterToCaloParticle_map));
}

void SimTrackstersProducer::returnEmptyCollections(edm::Event& evt, const int lcSize) {
  edm::LogWarning("SimTrackstersProducer") << "Missing input collections. Producing empty outputs.";
  // put into the event empty collections
  for (SimTsConfig const& config : simTsConfigs_) {
    evt.emplace(config.simTrackster_token);
    evt.emplace(config.outputMask_token, lcSize, 1.f);

    evt.emplace(config.simTracksterToSimCluster_map_token);
    evt.emplace(config.simTracksterToCaloParticle_map_token);
  }
}
