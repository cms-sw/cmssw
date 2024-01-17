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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
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
#include "RecoHGCal/TICL/plugins/GeneralInterpretationAlgo.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

#include "RecoHGCal/TICL/interface/GlobalCache.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"

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

  void energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                             const tensorflow::Session *,
                             std::vector<Trackster> &result) const;

  template <typename F>
  void assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates,
                              edm::Handle<std::vector<reco::Track>> track_h,
                              TICLInterpretationAlgoBase<reco::Track>::TrackTimingInformation inputTiming,
                              TrajTrackAssociationCollection trjtrks,
                              F func) const;

  std::unique_ptr<TICLInterpretationAlgoBase<reco::Track>> generalInterpretationAlgo_;
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> egamma_tracksters_tokens_;
  std::vector<edm::EDGetTokenT<std::vector<std::vector<unsigned>>>> egamma_tracksterlinks_tokens_;

  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> general_tracksters_tokens_;
  std::vector<edm::EDGetTokenT<std::vector<std::vector<unsigned>>>> general_tracksterlinks_tokens_;

  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;

  std::vector<edm::EDGetTokenT<std::vector<float>>> original_masks_tokens_;

  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_err_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_beta_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tracks_path_length_token_;
  edm::EDGetTokenT<edm::ValueMap<GlobalPoint>> tracks_glob_pos_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAssToken_;

  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const bool useMTDTiming_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const std::string detector_;
  const std::string propName_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

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
  const HGCalDDDConstants *hgcons_;
  hgcal::RecHitTools rhtools_;
  const float tkEnergyCut_ = 2.0f;
  const StringCutObjectSelector<reco::Track> cutTk_;
  static constexpr int eidNFeatures_ = 3;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
  edm::ESHandle<MagneticField> bfield_;
  edm::ESHandle<Propagator> propagator_;
  static constexpr float c_light_ = CLHEP::c_light * CLHEP::ns / CLHEP::cm;
};

TICLCandidateProducer::TICLCandidateProducer(const edm::ParameterSet &ps)
    : clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      trajTrackAssToken_(consumes<TrajTrackAssociationCollection>(ps.getParameter<edm::InputTag>("trjtrkAss"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      detector_(ps.getParameter<std::string>("detector")),
      propName_(ps.getParameter<std::string>("propagator")),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      bsToken_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamspot"))),
      tfDnnLabel_(ps.getParameter<std::string>("tfDnnLabel")),
      tfDnnToken_(esConsumes(edm::ESInputTag("", tfDnnLabel_))),
      tfSession_(nullptr),
      eidInputName_(ps.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(ps.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(ps.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(ps.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(ps.getParameter<int>("eid_n_layers")),
      eidNClusters_(ps.getParameter<int>("eid_n_clusters")),
      eidSession_(nullptr),
      cutTk_(ps.getParameter<std::string>("cutTk")) {
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
    std::string detectorName_ = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
    hdc_token_ = esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
        edm::ESInputTag("", detectorName_));
    tracks_time_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTime"));
    tracks_time_err_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeErr"));
    tracks_beta_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksBeta"));
    tracks_path_length_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksPathLength"));
    tracks_glob_pos_token_ =
        consumes<edm::ValueMap<GlobalPoint>>(ps.getParameter<edm::InputTag>("tracksGlobalPosition"));
  }

  produces<std::vector<TICLCandidate>>();

  // New trackster collection after linking
  produces<std::vector<Trackster>>();

  auto interpretationPSet = ps.getParameter<edm::ParameterSet>("interpretationDescPSet");
  auto algoType = interpretationPSet.getParameter<std::string>("type");
  generalInterpretationAlgo_ =
      TICLGeneralInterpretationPluginFactory::get()->create(algoType, interpretationPSet, consumesCollector());
}

void TICLCandidateProducer::beginJob() {}

void TICLCandidateProducer::endJob(){};

void TICLCandidateProducer::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
  hgcons_ = hdc.product();

  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);

  bfield_ = es.getHandle(bfield_token_);
  propagator_ = es.getHandle(propagator_token_);
  generalInterpretationAlgo_->initialize(hgcons_, rhtools_, bfield_, propagator_);
};

void filterTracks(edm::Handle<std::vector<reco::Track>> tkH,
                  const std::vector<reco::Muon> &muons,
                  const StringCutObjectSelector<reco::Track> cutTk_,
                  const float tkEnergyCut_,
                  std::vector<bool> &maskTracks) {
  auto const &tracks = *tkH;
  for (unsigned i = 0; i < tracks.size(); ++i) {
    const auto &tk = tracks[i];
    reco::TrackRef trackref = reco::TrackRef(tkH, i);

    // veto tracks associated to muons
    int muId = PFMuonAlgo::muAssocToTrack(trackref, muons);

    if (!cutTk_((tk)) or muId != -1) {
      maskTracks[i] = false;
      continue;
    }

    // don't consider tracks below 2 GeV for linking
    if (std::sqrt(tk.p() * tk.p() + ticl::mpion2) < tkEnergyCut_) {
      maskTracks[i] = false;
      continue;
    }

    // record tracks that can be used to make a ticlcandidate
    maskTracks[i] = true;
  }
}

void TICLCandidateProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  auto resultTracksters = std::make_unique<std::vector<Trackster>>();
  auto resultTrackstersMerged = std::make_unique<std::vector<Trackster>>();
  auto linkedResultTracksters = std::make_unique<std::vector<std::vector<unsigned int>>>();
  tfSession_ = es.getData(tfDnnToken_).getSession();

  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);
  auto const &muons = evt.get(muons_token_);

  edm::Handle<std::vector<reco::Track>> tracks_h;
  const auto &trjtrks = evt.get(trajTrackAssToken_);

  edm::Handle<edm::ValueMap<float>> trackTime_h;
  edm::Handle<edm::ValueMap<float>> trackTimeErr_h;
  edm::Handle<edm::ValueMap<float>> trackTimeBeta_h;
  edm::Handle<edm::ValueMap<float>> trackPathToMTD_h;
  edm::Handle<edm::ValueMap<GlobalPoint>> trackTimeGlobalPosition_h;
  evt.getByToken(tracks_token_, tracks_h);
  const auto &tracks = *tracks_h;
  if (useMTDTiming_) {
    evt.getByToken(tracks_time_token_, trackTime_h);
    evt.getByToken(tracks_time_err_token_, trackTimeErr_h);
    evt.getByToken(tracks_beta_token_, trackTimeBeta_h);
    evt.getByToken(tracks_path_length_token_, trackPathToMTD_h);
    evt.getByToken(tracks_glob_pos_token_, trackTimeGlobalPosition_h);
  }

  const auto &bs = evt.get(bsToken_);

  auto const bFieldProd = bfield_.product();
  const Propagator *propagator = propagator_.product();

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

  std::vector<bool> maskTracks;
  maskTracks.resize(tracks.size());
  filterTracks(tracks_h, muons, cutTk_, tkEnergyCut_, maskTracks);

  const typename TICLInterpretationAlgoBase<reco::Track>::Inputs input(evt,
                                                                       es,
                                                                       layerClusters,
                                                                       layerClustersTimes,
                                                                       generalTrackstersManager,
                                                                       generalTracksterLinksGlobalId,
                                                                       tracks_h,
                                                                       maskTracks);

  const typename TICLInterpretationAlgoBase<reco::Track>::TrackTimingInformation inputTiming(
      trackTime_h, trackTimeErr_h, trackTimeBeta_h, trackPathToMTD_h, trackTimeGlobalPosition_h);

  auto resultCandidates = std::make_unique<std::vector<TICLCandidate>>();
  std::vector<int> trackstersInTrackIndices(tracks.size(), -1);

  //TODO
  //egammaInterpretationAlg_->makecandidates(inputGSF, inputTiming, *resultTrackstersMerged, trackstersInGSFTrackIndices)
  // mask generalTracks associated to GSFTrack linked in egammaInterpretationAlgo_

  generalInterpretationAlgo_->makeCandidates(input, inputTiming, *resultTracksters, trackstersInTrackIndices);

  assignPCAtoTracksters(
      *resultTracksters, layerClusters, layerClustersTimes, rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(), true);

  energyRegressionAndID(layerClusters, tfSession_, *resultTracksters);

  std::vector<bool> maskTracksters(resultTracksters->size(), true);
  edm::OrphanHandle<std::vector<Trackster>> resultTracksters_h = evt.put(std::move(resultTracksters));
  //create ChargedCandidates
  for (size_t iTrack = 0; iTrack < tracks.size(); iTrack++) {
    auto const tracksterId = trackstersInTrackIndices[iTrack];
    auto trackPtr = edm::Ptr<reco::Track>(tracks_h, iTrack);
    if (tracksterId != -1 and !maskTracksters.empty()) {
      auto tracksterPtr = edm::Ptr<Trackster>(resultTracksters_h, tracksterId);
      TICLCandidate chargedCandidate(trackPtr, tracksterPtr);
      resultCandidates->push_back(chargedCandidate);
      maskTracksters[tracksterId] = false;
    } else {
      //charged candidates track only
      edm::Ptr<Trackster> tracksterPtr;
      TICLCandidate chargedCandidate(trackPtr, tracksterPtr);
      resultCandidates->push_back(chargedCandidate);
    }
  }

  //Neutral Candidate
  for (size_t iTrackster = 0; iTrackster < maskTracksters.size(); iTrackster++) {
    if (maskTracksters[iTrackster]) {
      edm::Ptr<Trackster> tracksterPtr(resultTracksters_h, iTrackster);
      edm::Ptr<reco::Track> trackPtr;
      TICLCandidate neutralCandidate(trackPtr, tracksterPtr);
      resultCandidates->push_back(neutralCandidate);
    }
  }

  auto getPathLength =
      [&](const reco::Track track, float zVal, const Trajectory &traj, TrajectoryStateClosestToBeamLine &tscbl) {
        TrajectoryStateOnSurface stateForProjectionToBeamLineOnSurface =
            traj.closestMeasurement(GlobalPoint(bs.x0(), bs.y0(), bs.z0())).updatedState();

        if (!stateForProjectionToBeamLineOnSurface.isValid()) {
          edm::LogError("CannotPropagateToBeamLine")
              << "the state on the closest measurement is not valid. skipping track.";
          return 0.f;
        }
        const FreeTrajectoryState &stateForProjectionToBeamLine = *stateForProjectionToBeamLineOnSurface.freeState();

        TSCBLBuilderWithPropagator tscblBuilder(*propagator);
        tscbl = tscblBuilder(stateForProjectionToBeamLine, bs);

        if (tscbl.isValid()) {
          auto const &tscblPCA = tscbl.trackStateAtPCA();
          auto const &innSurface = traj.direction() == alongMomentum ? traj.firstMeasurement().updatedState().surface()
                                                                     : traj.lastMeasurement().updatedState().surface();
          auto const &extSurface = traj.direction() == alongMomentum ? traj.lastMeasurement().updatedState().surface()
                                                                     : traj.firstMeasurement().updatedState().surface();
          float pathlength = propagator->propagateWithPath(tscblPCA, innSurface).second;

          if (pathlength) {
            const auto &fts_inn = trajectoryStateTransform::innerFreeState((track), bFieldProd);
            const auto &t_inn_out = propagator->propagateWithPath(fts_inn, extSurface);

            if (t_inn_out.first.isValid()) {
              pathlength += t_inn_out.second;

              std::pair<float, float> rMinMax = hgcons_->rangeR(zVal, true);

              int iSide = int(track.eta() > 0);
              float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
              const auto &disk = std::make_unique<GeomDet>(
                  Disk::build(Disk::PositionType(0, 0, zSide),
                              Disk::RotationType(),
                              SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                      .get());
              const auto &fts_out = trajectoryStateTransform::outerFreeState((track), bFieldProd);
              const auto &tsos = propagator->propagateWithPath(fts_out, disk->surface());

              if (tsos.first.isValid()) {
                pathlength += tsos.second;
                return pathlength;
              }
            }
          }
        }
        return 0.f;
      };

  assignTimeToCandidates(*resultCandidates, tracks_h, inputTiming, trjtrks, getPathLength);

  evt.put(std::move(resultCandidates));
}

void TICLCandidateProducer::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                  const tensorflow::Session *eidSession,
                                                  std::vector<Trackster> &tracksters) const {
  // Energy regression and particle identification strategy:
  //
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) threshold. Inference is not applied for soft tracksters.
  // 3. When no trackster passes the selection, return.
  // 4. Create input and output tensors. The batch dimension is determined by the number of
  //    selected tracksters.
  // 5. Fill input tensors with layer cluster features. Per layer, clusters are ordered descending
  //    by energy. Given that tensor data is contiguous in memory, we can use pointer arithmetic to
  //    fill values, even with batching.
  // 6. Zero-fill features for empty clusters in each layer.
  // 7. Batched inference.
  // 8. Assign the regressed energy and id probabilities to each trackster.
  //
  // Indices used throughout this method:
  // i -> batch element / trackster
  // j -> layer
  // k -> cluster
  // l -> feature

  // do nothing when no trackster passes the selection (3)
  int batchSize = (int)tracksters.size();
  if (batchSize == 0) {
    return;
  }

  for (auto &t : tracksters) {
    t.setRegressedEnergy(0.f);
    t.zeroProbabilities();
  }

  // create input and output tensors (4)
  tensorflow::TensorShape shape({batchSize, eidNLayers_, eidNClusters_, eidNFeatures_});
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  tensorflow::NamedTensorList inputList = {{eidInputName_, input}};
  static constexpr int inputDimension = 4;

  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> outputNames;
  if (!eidOutputNameEnergy_.empty()) {
    outputNames.push_back(eidOutputNameEnergy_);
  }
  if (!eidOutputNameId_.empty()) {
    outputNames.push_back(eidOutputNameId_);
  }

  // fill input tensor (5)
  for (int i = 0; i < batchSize; i++) {
    const Trackster &trackster = tracksters[i];

    // per layer, we only consider the first eidNClusters_ clusters in terms of
    // energy, so in order to avoid creating large / nested structures to do
    // the sorting for an unknown number of total clusters, create a sorted
    // list of layer cluster indices to keep track of the filled clusters
    std::vector<int> clusterIndices(trackster.vertices().size());
    for (int k = 0; k < (int)trackster.vertices().size(); k++) {
      clusterIndices[k] = k;
    }
    sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](const int &a, const int &b) {
      return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
    });

    // keep track of the number of seen clusters per layer
    std::vector<int> seenClusters(eidNLayers_);

    // loop through clusters by descending energy
    for (const int &k : clusterIndices) {
      // get features per layer and cluster and store the values directly in the input tensor
      const reco::CaloCluster &cluster = layerClusters[trackster.vertices(k)];
      int j = rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
      if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
        // get the pointer to the first feature value for the current batch, layer and cluster
        float *features = &input.tensor<float, inputDimension>()(i, j, seenClusters[j], 0);

        // fill features
        *(features++) = float(cluster.energy() / float(trackster.vertex_multiplicity(k)));
        *(features++) = float(std::abs(cluster.eta()));
        *(features) = float(cluster.phi());

        // increment seen clusters
        seenClusters[j]++;
      }
    }

    // zero-fill features of empty clusters in each layer (6)
    for (int j = 0; j < eidNLayers_; j++) {
      for (int k = seenClusters[j]; k < eidNClusters_; k++) {
        float *features = &input.tensor<float, inputDimension>()(i, j, k, 0);
        for (int l = 0; l < eidNFeatures_; l++) {
          *(features++) = 0.f;
        }
      }
    }
  }

  // run the inference (7)
  tensorflow::run(eidSession, inputList, outputNames, &outputs);

  // store regressed energy per trackster (8)
  if (!eidOutputNameEnergy_.empty()) {
    // get the pointer to the energy tensor, dimension is batch x 1
    float *energy = outputs[0].flat<float>().data();

    for (int i = 0; i < batchSize; ++i) {
      float regressedEnergy =
          tracksters[i].raw_energy() > eidMinClusterEnergy_ ? energy[i] : tracksters[i].raw_energy();
      tracksters[i].setRegressedEnergy(regressedEnergy);
    }
  }

  // store id probabilities per trackster (8)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    int probsIdx = !eidOutputNameEnergy_.empty();
    float *probs = outputs[probsIdx].flat<float>().data();
    int probsNumber = tracksters[0].id_probabilities().size();
    for (int i = 0; i < batchSize; ++i) {
      tracksters[i].setProbabilities(&probs[i * probsNumber]);
    }
  }
}

template <typename F>
void TICLCandidateProducer::assignTimeToCandidates(
    std::vector<TICLCandidate> &resultCandidates,
    edm::Handle<std::vector<reco::Track>> track_h,
    TICLInterpretationAlgoBase<reco::Track>::TrackTimingInformation inputTiming,
    TrajTrackAssociationCollection trjtrks,
    F func) const {
  for (auto &cand : resultCandidates) {
    float beta = 1;
    float time = 0.f;
    float invTimeErr = 0.f;

    // if (not cand.tracksters().size())
    //  continue;
    for (const auto &tr : cand.tracksters()) {
      if (tr->timeError() > 0) {
        const auto invTimeESq = pow(tr->timeError(), -2);
        const auto x = tr->barycenter().X();
        const auto y = tr->barycenter().Y();
        const auto z = tr->barycenter().Z();
        auto path = std::sqrt(x * x + y * y + z * z);
        if (cand.trackPtr().get() != nullptr) {
          const auto &trackIndex = cand.trackPtr().get() - (edm::Ptr<reco::Track>(track_h, 0)).get();
          const auto &trackRef = edm::Ref<std::vector<reco::Track>>(track_h, trackIndex);
          if (useMTDTiming_ and (*inputTiming.tkTimeErr_h)[trackRef] > 0) {
            const auto &trackMtdPos = (*inputTiming.tkMtdPos_h);
            const auto xMtd = trackMtdPos[trackRef].x();
            const auto yMtd = trackMtdPos[trackRef].y();
            const auto zMtd = trackMtdPos[trackRef].z();

            beta = (*inputTiming.tkBeta_h)[trackRef];
            path = std::sqrt((x - xMtd) * (x - xMtd) + (y - yMtd) * (y - yMtd) + (z - zMtd) * (z - zMtd)) +
                   (*inputTiming.tkPath_h)[trackRef];
          } else {
            const auto &trackIndex = cand.trackPtr().get() - (edm::Ptr<reco::Track>(track_h, 0)).get();
            for (const auto &trj : trjtrks) {
              if (trj.val != edm::Ref<std::vector<reco::Track>>(track_h, trackIndex))
                continue;
              const Trajectory &traj = *trj.key;
              TrajectoryStateClosestToBeamLine tscbl;

              float pathLength = func(*(cand.trackPtr().get()), z, traj, tscbl);
              if (pathLength) {
                path = pathLength;
                break;
              }
            }
          }
        }
        time += (tr->time() - path / (beta * c_light_)) * invTimeESq;
        invTimeErr += invTimeESq;
      }
    }
    if (invTimeErr > 0) {
      cand.setTime(time / invTimeErr);
      // FIXME_ set a liminf of 0.02 ns on the ts error (based on residuals)
      auto timeErr = sqrt(1.f / invTimeErr) > 0.02 ? sqrt(1.f / invTimeErr) : 0.02;
      cand.setTimeError(timeErr);
    }
  }
}

void TICLCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription interpretationDesc;
  interpretationDesc.addNode(edm::PluginDescription<TICLGeneralInterpretationPluginFactory>("type", "General", true));
  desc.add<edm::ParameterSetDescription>("interpretationDescPSet", interpretationDesc);
  desc.add<std::vector<edm::InputTag>>("egamma_tracksters_collections", {edm::InputTag("ticlTracksterLinks")});
  desc.add<std::vector<edm::InputTag>>("egamma_tracksterlinks_collections", {edm::InputTag("ticlTracksterLinks")});
  desc.add<std::vector<edm::InputTag>>("general_tracksters_collections", {edm::InputTag("ticlTracksterLinks")});
  desc.add<std::vector<edm::InputTag>>("general_tracksterlinks_collections", {edm::InputTag("ticlTracksterLinks")});
  desc.add<std::vector<edm::InputTag>>("original_masks",
                                       {edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask")});
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("trjtrkAss", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("tracksTime", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("tracksTimeErr", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("tracksBeta", edm::InputTag("trackExtenderWithMTD:generalTrackBeta"));
  desc.add<edm::InputTag>("tracksGlobalPosition", edm::InputTag("trackExtenderWithMTD:generalTrackmtdpos"));
  desc.add<edm::InputTag>("tracksPathLength", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<edm::InputTag>("beamspot", edm::InputTag("offlineBeamSpot"));
  desc.add<bool>("useMTDTiming", true);
  desc.add<std::string>("tfDnnLabel", "tracksterSelectionTf");
  desc.add<std::string>("eid_input_name", "input");
  desc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
  desc.add<std::string>("eid_output_name_id", "output/id_probabilities");
  desc.add<double>("eid_min_cluster_energy", 2.5);
  desc.add<int>("eid_n_layers", 50);
  desc.add<int>("eid_n_clusters", 10);
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  descriptions.add("ticlCandidateProducer", desc);
}

DEFINE_FWK_MODULE(TICLCandidateProducer);
