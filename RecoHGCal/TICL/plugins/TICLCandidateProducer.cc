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
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "RecoHGCal/TICL/plugins/TICLInterpretationPluginFactory.h"
#include "RecoHGCal/TICL/plugins/GeneralInterpretationAlgo.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

#include "RecoHGCal/TICL/interface/GlobalCache.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

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

  void beginRun(edm::Run const &iEvent, edm::EventSetup const &es) override;

private:
  void dumpCandidate(const TICLCandidate &) const;

  template <typename F>
  void assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates,
                              edm::Handle<std::vector<reco::Track>> track_h,
                              MtdHostCollection::ConstView &inputTimingView,
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
  edm::EDGetTokenT<MtdHostCollection> inputTimingToken_;

  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const bool useMTDTiming_;
  const bool useTimingAverage_;
  const float timingQualityThreshold_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const std::string detector_;
  const std::string propName_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> trackingGeometry_token_;

  std::once_flag initializeGeometry_;
  const HGCalDDDConstants *hgcons_;
  hgcal::RecHitTools rhtools_;
  const float tkEnergyCut_ = 2.0f;
  const StringCutObjectSelector<reco::Track> cutTk_;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
  edm::ESHandle<MagneticField> bfield_;
  edm::ESHandle<Propagator> propagator_;
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry_;
  static constexpr float c_light_ = CLHEP::c_light * CLHEP::ns / CLHEP::cm;
  static constexpr float timeRes = 0.02f;
};

TICLCandidateProducer::TICLCandidateProducer(const edm::ParameterSet &ps)
    : clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      useMTDTiming_(ps.getParameter<bool>("useMTDTiming")),
      useTimingAverage_(ps.getParameter<bool>("useTimingAverage")),
      timingQualityThreshold_(ps.getParameter<double>("timingQualityThreshold")),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      detector_(ps.getParameter<std::string>("detector")),
      propName_(ps.getParameter<std::string>("propagator")),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      trackingGeometry_token_(
          esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord, edm::Transition::BeginRun>()),
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

  std::string detectorName_ = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
  hdc_token_ =
      esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", detectorName_));
  if (useMTDTiming_) {
    inputTimingToken_ = consumes<MtdHostCollection>(ps.getParameter<edm::InputTag>("timingSoA"));
  }

  produces<std::vector<TICLCandidate>>();

  // New trackster collection after linking
  produces<std::vector<Trackster>>();

  auto interpretationPSet = ps.getParameter<edm::ParameterSet>("interpretationDescPSet");
  auto algoType = interpretationPSet.getParameter<std::string>("type");
  generalInterpretationAlgo_ =
      TICLGeneralInterpretationPluginFactory::get()->create(algoType, interpretationPSet, consumesCollector());
}

void TICLCandidateProducer::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
  hgcons_ = hdc.product();

  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);

  bfield_ = es.getHandle(bfield_token_);
  propagator_ = es.getHandle(propagator_token_);
  generalInterpretationAlgo_->initialize(hgcons_, rhtools_, bfield_, propagator_);

  trackingGeometry_ = es.getHandle(trackingGeometry_token_);
};

void filterTracks(edm::Handle<std::vector<reco::Track>> tkH,
                  const edm::Handle<std::vector<reco::Muon>> &muons_h,
                  const StringCutObjectSelector<reco::Track> cutTk_,
                  const float tkEnergyCut_,
                  std::vector<bool> &maskTracks) {
  auto const &tracks = *tkH;
  for (unsigned i = 0; i < tracks.size(); ++i) {
    const auto &tk = tracks[i];
    reco::TrackRef trackref = reco::TrackRef(tkH, i);

    // veto tracks associated to muons
    int muId = PFMuonAlgo::muAssocToTrack(trackref, *muons_h);
    const reco::MuonRef muonref = reco::MuonRef(muons_h, muId);

    if (!cutTk_((tk)) or (muId != -1 and PFMuonAlgo::isMuon(muonref) and not(*muons_h)[muId].isTrackerMuon())) {
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

  const auto &layerClusters = evt.get(clusters_token_);
  const auto &layerClustersTimes = evt.get(clustersTime_token_);
  edm::Handle<reco::MuonCollection> muons_h;
  evt.getByToken(muons_token_, muons_h);

  edm::Handle<std::vector<reco::Track>> tracks_h;
  evt.getByToken(tracks_token_, tracks_h);
  const auto &tracks = *tracks_h;

  edm::Handle<MtdHostCollection> inputTiming_h;
  MtdHostCollection::ConstView inputTimingView;
  if (useMTDTiming_) {
    evt.getByToken(inputTimingToken_, inputTiming_h);
    inputTimingView = (*inputTiming_h).const_view();
  }

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
  filterTracks(tracks_h, muons_h, cutTk_, tkEnergyCut_, maskTracks);

  const typename TICLInterpretationAlgoBase<reco::Track>::Inputs input(evt,
                                                                       es,
                                                                       layerClusters,
                                                                       layerClustersTimes,
                                                                       generalTrackstersManager,
                                                                       generalTracksterLinksGlobalId,
                                                                       tracks_h,
                                                                       maskTracks);

  auto resultCandidates = std::make_unique<std::vector<TICLCandidate>>();
  std::vector<int> trackstersInTrackIndices(tracks.size(), -1);

  //TODO
  //egammaInterpretationAlg_->makecandidates(inputGSF, inputTiming, *resultTrackstersMerged, trackstersInGSFTrackIndices)
  // mask generalTracks associated to GSFTrack linked in egammaInterpretationAlgo_

  generalInterpretationAlgo_->makeCandidates(input, inputTiming_h, *resultTracksters, trackstersInTrackIndices);

  assignPCAtoTracksters(*resultTracksters,
                        layerClusters,
                        layerClustersTimes,
                        rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z(),
                        rhtools_,
                        true);

  std::vector<bool> maskTracksters(resultTracksters->size(), true);
  edm::OrphanHandle<std::vector<Trackster>> resultTracksters_h = evt.put(std::move(resultTracksters));
  //create ChargedCandidates
  for (size_t iTrack = 0; iTrack < tracks.size(); iTrack++) {
    if (maskTracks[iTrack]) {
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
        auto trackRef = edm::Ref<reco::TrackCollection>(tracks_h, iTrack);
        const int muId = PFMuonAlgo::muAssocToTrack(trackRef, *muons_h);
        const reco::MuonRef muonRef = reco::MuonRef(muons_h, muId);
        if (muonRef.isNonnull() and muonRef->isGlobalMuon()) {
          // create muon candidate
          chargedCandidate.setPdgId(13 * trackPtr.get()->charge());
        }
        resultCandidates->push_back(chargedCandidate);
      }
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
      [&](const reco::Track &track, float zVal) {
        const auto &fts_inn = trajectoryStateTransform::innerFreeState(track, bFieldProd);
        const auto &fts_out = trajectoryStateTransform::outerFreeState(track, bFieldProd);
        const auto &surf_inn = trajectoryStateTransform::innerStateOnSurface(track, *trackingGeometry_, bFieldProd);
        const auto &surf_out = trajectoryStateTransform::outerStateOnSurface(track, *trackingGeometry_, bFieldProd);

        Basic3DVector<float> pos(track.referencePoint());
        Basic3DVector<float> mom(track.momentum());
        FreeTrajectoryState stateAtBeamspot{GlobalPoint(pos), GlobalVector(mom), track.charge(), bFieldProd};

        float pathlength = propagator->propagateWithPath(stateAtBeamspot, surf_inn.surface()).second;

        if (pathlength) {
          const auto &t_inn_out = propagator->propagateWithPath(fts_inn, surf_out.surface());

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
            const auto &tsos = propagator->propagateWithPath(fts_out, disk->surface());

            if (tsos.first.isValid()) {
              pathlength += tsos.second;
              return pathlength;
            }
          }
        }
#ifdef EDM_ML_DEBUG
        LogDebug("TICLCandidateProducer")
            << "Not able to use the track to compute the path length. A straight line will be used instead.";
#endif
        return 0.f;
      };

  assignTimeToCandidates(*resultCandidates, tracks_h, inputTimingView, getPathLength);

  evt.put(std::move(resultCandidates));
}

template <typename F>
void TICLCandidateProducer::assignTimeToCandidates(std::vector<TICLCandidate> &resultCandidates,
                                                   edm::Handle<std::vector<reco::Track>> track_h,
                                                   MtdHostCollection::ConstView &inputTimingView,
                                                   F func) const {
  for (auto &cand : resultCandidates) {
    float beta = 1;
    float time = 0.f;
    float invTimeErr = 0.f;
    float timeErr = -1.f;

    for (const auto &tr : cand.tracksters()) {
      if (tr->timeError() > 0) {
        const auto invTimeESq = pow(tr->timeError(), -2);
        const auto x = tr->barycenter().X();
        const auto y = tr->barycenter().Y();
        const auto z = tr->barycenter().Z();
        auto path = std::sqrt(x * x + y * y + z * z);
        if (cand.trackPtr().get() != nullptr) {
          const auto &trackIndex = cand.trackPtr().get() - (edm::Ptr<reco::Track>(track_h, 0)).get();
          if (useMTDTiming_ and inputTimingView.timeErr()[trackIndex] > 0) {
            const auto xMtd = inputTimingView.posInMTD_x()[trackIndex];
            const auto yMtd = inputTimingView.posInMTD_y()[trackIndex];
            const auto zMtd = inputTimingView.posInMTD_z()[trackIndex];

            beta = inputTimingView.beta()[trackIndex];
            path = std::sqrt((x - xMtd) * (x - xMtd) + (y - yMtd) * (y - yMtd) + (z - zMtd) * (z - zMtd)) +
                   inputTimingView.pathLength()[trackIndex];
          } else {
            float pathLength = func(*(cand.trackPtr().get()), z);
            if (pathLength) {
              path = pathLength;
            }
          }
        }
        time += (tr->time() - path / (beta * c_light_)) * invTimeESq;
        invTimeErr += invTimeESq;
      }
    }
    if (invTimeErr > 0) {
      time = time / invTimeErr;
      // FIXME_ set a liminf of 0.02 ns on the ts error (based on residuals)
      timeErr = sqrt(1.f / invTimeErr);
      if (timeErr < timeRes)
        timeErr = timeRes;
      cand.setTime(time, timeErr);
    }

    if (useMTDTiming_ and cand.charge()) {
      // Check MTD timing availability
      const auto &trackIndex = cand.trackPtr().get() - (edm::Ptr<reco::Track>(track_h, 0)).get();
      const bool assocQuality = inputTimingView.MVAquality()[trackIndex] > timingQualityThreshold_;
      if (assocQuality) {
        const auto timeHGC = cand.time();
        const auto timeEHGC = cand.timeError();
        const auto timeMTD = inputTimingView.time0()[trackIndex];
        const auto timeEMTD = inputTimingView.time0Err()[trackIndex];

        if (useTimingAverage_ && (timeEMTD > 0 && timeEHGC > 0)) {
          // Compute weighted average between HGCAL and MTD timing
          const auto invTimeESqHGC = pow(timeEHGC, -2);
          const auto invTimeESqMTD = pow(timeEMTD, -2);
          timeErr = 1.f / (invTimeESqHGC + invTimeESqMTD);
          time = (timeHGC * invTimeESqHGC + timeMTD * invTimeESqMTD) * timeErr;
          timeErr = sqrt(timeErr);
        } else if (timeEMTD > 0) {
          time = timeMTD;
          timeErr = timeEMTD;
        }
      }
      cand.setTime(time, timeErr);
      cand.setMTDTime(inputTimingView.time()[trackIndex], inputTimingView.timeErr()[trackIndex]);
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
  desc.add<edm::InputTag>("timingSoA", edm::InputTag("mtdSoA"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<bool>("useMTDTiming", true);
  desc.add<bool>("useTimingAverage", true);
  desc.add<double>("timingQualityThreshold", 0.5);
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  descriptions.add("ticlCandidateProducer", desc);
}

DEFINE_FWK_MODULE(TICLCandidateProducer);
