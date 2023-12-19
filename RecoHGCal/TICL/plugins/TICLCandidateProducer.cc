// Author: Felice Pantaleo, Wahid Redjeb, Aurora Perego (CERN) - felice.pantaleo@cern.ch, wahid.redjeb@cern.ch, aurora.perego@cern.ch
// Date: 12/2023
#include <memory>  // unique_ptr
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "RecoHGCal/TICL/plugins/TICLInterpretationPluginFactory.h"

#include "RecoHGCal/TICL/interface/GlobalCache.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackstersPCA.h"

using namespace ticl;

class TICLCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit TICLCandidateProducer(const edm::ParameterSet &ps);
  ~TICLCandidateProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob();
  void endJob();

  void beginRun(edm::Run const &iEvent, edm::EventSetup const &es) override;

private:
  void dumpCandidate(const TICLCandidate &) const;

  // std::unique_ptr<InterpretationAlgoBase> interpretationAlgo_;
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> egamma_tracksters_tokens_;
  std::vector<edm::EDGetTokenT<std::vector<std::vector<unsigned>>>> egamma_tracksterlinks_tokens_;

  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> general_tracksters_tokens_;
  std::vector<edm::EDGetTokenT<std::vector<std::vector<unsigned>>>> general_tracksterlinks_tokens_;

  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;

  std::vector<edm::EDGetTokenT<std::vector<float>>> original_masks_tokens_;

  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_quality_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_err_token_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const bool useMTDTiming_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const std::string propName_;


  const std::string tfDnnLabel_;
  const edm::ESGetToken<TfGraphDefWrapper, TfGraphRecord> tfDnnToken_;
  const tensorflow::Session *tfSession_;
  const std::string eidInputName_;
  const std::string eidOutputNameEnergy_;
  const std::string eidOutputNameId_;
  const float eidMinClusterEnergy_;
  const int eidNLayers_;
  const int eidNClusters_;
  tensorflow::Session *eidSession_;

  std::once_flag initializeGeometry_;
  hgcal::RecHitTools rhtools_;
};

TICLCandidateProducer::TICLCandidateProducer(const edm::ParameterSet &ps)
    : clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      propName_(ps.getParameter<std::string>("propagator")),
      tfDnnLabel_(ps.getParameter<std::string>("tfDnnLabel")),
      tfDnnToken_(esConsumes(edm::ESInputTag("", tfDnnLabel_))),
      tfSession_(nullptr),
      eidInputName_(ps.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(ps.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(ps.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(ps.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(ps.getParameter<int>("eid_n_layers")),
      eidNClusters_(ps.getParameter<int>("eid_n_clusters")),
      eidSession_(nullptr) {
  // These are the CLUE3DEM Tracksters put in the event by the TracksterLinksProducer with the superclustering algorithm
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("egamma_tracksters_collections")) {
    egamma_tracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
  }

  // These are the links put in the event by the TracksterLinksProducer with the superclustering algorithm
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("egamma_tracksterlinks_collections")) {
    egamma_tracksterlinks_tokens_.emplace_back(consumes<std::vector<std::vector<unsigned int>>>(tag));
  }

  //make sure that the number of tracksters collections and tracksterlinks collections is the same
  assert(egamma_tracksters_tokens_.size() == egamma_tracksterlinks_tokens_.size());

  // Loop over the edm::VInputTag and append the token to general_tracksters_tokens_
  // These, instead, are the tracksters already merged by the TrackstersLinksProducer
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("general_tracksters_collections")) {
    general_tracksters_tokens_.emplace_back(consumes<std::vector<Trackster>>(tag));
  }

  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("general_tracksterlinks_collections")) {
    general_tracksterlinks_tokens_.emplace_back(consumes<std::vector<std::vector<unsigned int>>>(tag));
  }

  //make sure that the number of tracksters collections and tracksterlinks collections is the same
  assert(general_tracksters_tokens_.size() == general_tracksterlinks_tokens_.size());

  //Loop over the edm::VInputTag of masks and append the token to original_masks_tokens_
  for (auto const &tag : ps.getParameter<std::vector<edm::InputTag>>("original_masks")) {
    original_masks_tokens_.emplace_back(consumes<std::vector<float>>(tag));
  }

  if (useMTDTiming_) {
    tracks_time_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTime"));
    tracks_time_quality_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeQual"));
    tracks_time_err_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeErr"));
  }

  produces<std::vector<TICLCandidate>>();

  // New trackster collection after linking
  produces<std::vector<Trackster>>();

  // auto linkingPSet = ps.getParameter<edm::ParameterSet>("linkingPSet");
  // auto algoType = linkingPSet.getParameter<std::string>("type");
  // linkingAlgo_ = TracksterLinkingPluginFactory::get()->create(algoType, linkingPSet, consumesCollector());
}

void TICLCandidateProducer::beginJob() {}

void TICLCandidateProducer::endJob(){};

void TICLCandidateProducer::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);
};

void TICLCandidateProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  auto resultTracksters = std::make_unique<std::vector<Trackster>>();
  auto resultTrackstersMerged = std::make_unique<std::vector<Trackster>>();

  auto linkedResultTracksters = std::make_unique<std::vector<std::vector<unsigned int>>>();

  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);

  edm::Handle<std::vector<reco::Track>> track_h;

  edm::Handle<edm::ValueMap<float>> trackTime_h;
  edm::Handle<edm::ValueMap<float>> trackTimeErr_h;
  edm::Handle<edm::ValueMap<float>> trackTimeQual_h;
  evt.getByToken(tracks_token_, track_h);
  const auto &tracks = *track_h;
  if (useMTDTiming_) {
    evt.getByToken(tracks_time_token_, trackTime_h);
    evt.getByToken(tracks_time_err_token_, trackTimeErr_h);
    evt.getByToken(tracks_time_quality_token_, trackTimeQual_h);
  }

  // loop over the original_masks_tokens_ and get the original masks collections and multiply them
  // to get the global mask
  std::vector<float> original_global_mask(layerClusters.size(), 1.f);
  for (unsigned int i = 0; i < original_masks_tokens_.size(); ++i) {
    const auto &tmp_mask = evt.get(original_masks_tokens_[i]);
    for (unsigned int j = 0; j < tmp_mask.size(); ++j) {
      original_global_mask[j] *= tmp_mask[j];
    }
  }

  auto resultMask = std::make_unique<std::vector<float>>(original_global_mask);

  std::vector<edm::Handle<std::vector<Trackster>>> general_tracksters_h(general_tracksters_tokens_.size());
  MultiVectorManager<Trackster> generalTrackstersManager;
  for (unsigned int i = 0; i < general_tracksters_tokens_.size(); ++i) {
    evt.getByToken(general_tracksters_tokens_[i], general_tracksters_h[i]);
    //Fill MultiVectorManager
    generalTrackstersManager.addVector(*general_tracksters_h[i]);
  }
  //now get the general_tracksterlinks_tokens_
  std::vector<edm::Handle<std::vector<std::vector<unsigned>>>> general_tracksterlinks_h(
      general_tracksterlinks_tokens_.size());
  std::vector<std::vector<unsigned>> generalTracksterLinksGlobalId;
  for (unsigned int i = 0; i < general_tracksterlinks_tokens_.size(); ++i) {
    evt.getByToken(general_tracksterlinks_tokens_[i], general_tracksterlinks_h[i]);
    for (unsigned int j = 0; j < general_tracksterlinks_h[i]->size(); ++j) {
      generalTracksterLinksGlobalId.emplace_back();
      auto &links_vector = generalTracksterLinksGlobalId.back();
      links_vector.resize((*general_tracksterlinks_h[i])[j].size());
      for (unsigned int k = 0; k < links_vector.size(); ++k) {
        links_vector[k] = generalTrackstersManager.getGlobalIndex(i, (*general_tracksterlinks_h[i])[j][k]);
      }
    }
  }
  const typename TICLInterpretationAlgoBase<reco::Track>::Inputs input(
      evt, es, layerClusters, layerClustersTimes, generalTrackstersManager, generalTracksterLinksGlobalId, tracks);
}

void TICLCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription linkingDesc;
  // linkingDesc.addNode(edm::PluginDescription<TracksterLinkingPluginFactory>("type", "FastJet", true));
  desc.add<edm::ParameterSetDescription>("linkingPSet", linkingDesc);
  desc.add<std::vector<edm::InputTag>>(
      "tracksters_collections", {edm::InputTag("ticlTrackstersCLUE3DEM"), edm::InputTag("ticlTrackstersCLUE3DHAD")});
  desc.add<std::vector<edm::InputTag>>("original_masks",
                                       {edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask")});
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  descriptions.add("ticlCandidateProducer", desc);
}

DEFINE_FWK_MODULE(TICLCandidateProducer);
