#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "RecoTracker/MkFit/interface/MkFitEventOfHits.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit indludes
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

namespace {
  template <typename T>
  bool isPhase1Barrel(T subdet) {
    return subdet == PixelSubdetector::PixelBarrel || subdet == StripSubdetector::TIB ||
           subdet == StripSubdetector::TOB;
  }

  template <typename T>
  bool isPhase1Endcap(T subdet) {
    return subdet == PixelSubdetector::PixelEndcap || subdet == StripSubdetector::TID ||
           subdet == StripSubdetector::TEC;
  }
}  // namespace

class MkFitOutputTrackConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitOutputTrackConverter(edm::ParameterSet const& iConfig);
  ~MkFitOutputTrackConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  void convertCandidates(const MkFitOutputWrapper& mkFitOutput,
                         const mkfit::EventOfHits& eventOfHits,
                         const MkFitClusterIndexToHit& pixelClusterIndexToHit,
                         const MkFitClusterIndexToHit& stripClusterIndexToHit,
                         const edm::View<TrajectorySeed>& seeds,
                         const MagneticField& mf,
                         const Propagator& propagatorAlong,
                         const Propagator& propagatorOpposite,
                         const MkFitGeometry& mkFitGeom,
                         const TrackerTopology& tTopo,
                         const TkClonerImpl& hitCloner,
                         const std::vector<const DetLayer*>& detLayers,
                         const mkfit::TrackVec& mkFitSeeds,
                         const reco::BeamSpot* bs,
                         const NavigationSchool& navSchool,
                         const MeasurementTrackerEvent& measTk,
                         reco::TrackCollection& trks,
                         std::vector<int>& seedIndices,
                         std::vector<edm::OwnVector<TrackingRecHit>>& hitsVecs) const;

  std::pair<TrajectoryStateOnSurface, const GeomDet*> convertInnermostState(const FreeTrajectoryState& fts,
                                                                            const edm::OwnVector<TrackingRecHit>& hits,
                                                                            const Propagator& propagatorAlong,
                                                                            const Propagator& propagatorOpposite) const;
  float ptcorr(const float abstheta, const float pt) const;

  const edm::EDGetTokenT<MkFitEventOfHits> eventOfHitsToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> stripClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitSeedWrapper> mkfitSeedToken_;
  const edm::EDGetTokenT<MkFitOutputWrapper> tracksToken_;
  const edm::EDGetTokenT<edm::View<TrajectorySeed>> seedToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDPutTokenT<reco::TrackCollection> putTrackToken_;
  const edm::EDPutTokenT<TrackingRecHitCollection> putHitsToken_;
  const edm::EDPutTokenT<reco::TrackExtraCollection> putExtraToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> putSeedStopInfoToken_;

  const float qualityMaxInvPt_;
  const float qualityMinTheta_;
  const float qualityMaxRsq_;
  const float qualityMaxZ_;
  const float qualityMaxPosErrSq_;
  const bool qualitySignPt_;

  const bool calibrate_;
  std::vector<double> calibBinCenter_;
  std::vector<double> calibBinCoeff_;
  std::vector<double> calibBinOffset_;

  const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventToken_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navToken_;

  const int algo_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
};

MkFitOutputTrackConverter::MkFitOutputTrackConverter(edm::ParameterSet const& iConfig)
    : eventOfHitsToken_{consumes<MkFitEventOfHits>(iConfig.getParameter<edm::InputTag>("mkFitEventOfHits"))},
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
      stripClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitStripHits"))},
      mkfitSeedToken_{consumes<MkFitSeedWrapper>(iConfig.getParameter<edm::InputTag>("mkFitSeeds"))},
      tracksToken_{consumes<MkFitOutputWrapper>(iConfig.getParameter<edm::InputTag>("src"))},
      seedToken_{consumes<edm::View<TrajectorySeed>>(iConfig.getParameter<edm::InputTag>("seeds"))},
      propagatorAlongToken_{
          esConsumes<Propagator, TrackingComponentsRecord>(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))},
      propagatorOppositeToken_{esConsumes<Propagator, TrackingComponentsRecord>(
          iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))},
      mfToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      tTopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      putSeedStopInfoToken_{produces<std::vector<SeedStopInfo>>()},
      qualityMaxInvPt_{float(iConfig.getParameter<double>("qualityMaxInvPt"))},
      qualityMinTheta_{float(iConfig.getParameter<double>("qualityMinTheta"))},
      qualityMaxRsq_{float(pow(iConfig.getParameter<double>("qualityMaxR"), 2))},
      qualityMaxZ_{float(iConfig.getParameter<double>("qualityMaxZ"))},
      qualityMaxPosErrSq_{float(pow(iConfig.getParameter<double>("qualityMaxPosErr"), 2))},
      qualitySignPt_{iConfig.getParameter<bool>("qualitySignPt")},
      calibrate_{iConfig.getParameter<bool>("calibrate")},
      calibBinCenter_{iConfig.getParameter<std::vector<double>>("calibBinCenter")},
      calibBinCoeff_{iConfig.getParameter<std::vector<double>>("calibBinCoeff")},
      calibBinOffset_{iConfig.getParameter<std::vector<double>>("calibBinOffset")},
      measurementTrackerEventToken_{consumes(iConfig.getParameter<edm::InputTag>("measurementTrackerEvent"))},
      navToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("NavigationSchool"))},
      algo_{reco::TrackBase::algoByName(
          TString(iConfig.getParameter<edm::InputTag>("seeds").label()).ReplaceAll("Seeds", "").Data())},
      bsToken_(consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))) {
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}

void MkFitOutputTrackConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("mkFitEventOfHits", edm::InputTag{"mkFitEventOfHits"});
  desc.add("mkFitPixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("mkFitStripHits", edm::InputTag{"mkFitSiStripHits"});
  desc.add("mkFitSeeds", edm::InputTag{"mkFitSeedConverter"});
  desc.add("src", edm::InputTag{"mkFitProducer"});
  desc.add("seeds", edm::InputTag{"initialStepSeeds"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});
  desc.add("propagatorAlong", edm::ESInputTag{"", "PropagatorWithMaterial"});
  desc.add("propagatorOpposite", edm::ESInputTag{"", "PropagatorWithMaterialOpposite"});

  desc.add<double>("qualityMaxInvPt", 100)->setComment("max(1/pt) for converted tracks");
  desc.add<double>("qualityMinTheta", 0.01)->setComment("lower bound on theta (or pi-theta) for converted tracks");
  desc.add<double>("qualityMaxR", 120)->setComment("max(R) for the state position for converted tracks");
  desc.add<double>("qualityMaxZ", 280)->setComment("max(|Z|) for the state position for converted tracks");
  desc.add<double>("qualityMaxPosErr", 100)->setComment("max position error for converted tracks");
  desc.add<bool>("qualitySignPt", true)->setComment("check sign of 1/pt for converted tracks");

  desc.add<bool>("calibrate", true)->setComment("true if mkFit fit pT calibration is needed");
  desc.add<std::vector<double>>("calibBinCenter", {0.1704, 0.6028, 1.0188, 1.2898, 1.4390, 1.4908, 1.5500})
      ->setComment("calibration bin center (in |theta|)");
  desc.add<std::vector<double>>("calibBinCoeff", {1.0000, 1.0004, 1.00014, 1.0027, 1.0029, 1.0009, 0.9999})
      ->setComment("calibration coeff for bin");
  desc.add<std::vector<double>>("calibBinOffset", {0.0016, 0.0032, 0.0033, 0.0045, 0.0005, 0.0012, 0.0003})
      ->setComment("calibration offset for bin");

  desc.add<edm::ESInputTag>("NavigationSchool", edm::ESInputTag{"", "SimpleNavigationSchool"});
  desc.add<edm::InputTag>("measurementTrackerEvent", edm::InputTag("MeasurementTrackerEvent"));

  descriptions.addWithDefaultLabel(desc);
}

void MkFitOutputTrackConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<edm::View<TrajectorySeed>> hseeds;
  iEvent.getByToken(seedToken_, hseeds);
  const auto& seeds = *hseeds;
  const auto& mkfitSeeds = iEvent.get(mkfitSeedToken_);

  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto* tkBuilder = dynamic_cast<TkTransientTrackingRecHitBuilder const*>(&ttrhBuilder);
  if (!tkBuilder) {
    throw cms::Exception("LogicError") << "TTRHBuilder must be of type TkTransientTrackingRecHitBuilder";
  }
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);
  const auto& navSchool = iSetup.getData(navToken_);

  //const MeasurementTrackerEvent* measurementTracker;
  //if (!measurementTrackerEventToken_.isUninitialized()) {
  edm::Handle<MeasurementTrackerEvent> hmte;
  iEvent.getByToken(measurementTrackerEventToken_, hmte);
  const MeasurementTrackerEvent* measurementTracker = hmte.product();
  //}

  //beamspot for trk
  const reco::BeamSpot* beamspot = &iEvent.get(bsToken_);

  std::unique_ptr<reco::TrackCollection> trks(new reco::TrackCollection);
  std::unique_ptr<TrackingRecHitCollection> hits(new TrackingRecHitCollection());
  std::unique_ptr<reco::TrackExtraCollection> extras(new reco::TrackExtraCollection());

  std::vector<int> seedIndices;
  std::vector<edm::OwnVector<TrackingRecHit>> hitsVecs;

  // product references
  reco::TrackExtraRefProd ref_trackextras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  TrackingRecHitRefProd ref_rechits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;

  convertCandidates(iEvent.get(tracksToken_),
                    iEvent.get(eventOfHitsToken_).get(),
                    iEvent.get(pixelClusterIndexToHitToken_),
                    iEvent.get(stripClusterIndexToHitToken_),
                    seeds,
                    iSetup.getData(mfToken_),
                    iSetup.getData(propagatorAlongToken_),
                    iSetup.getData(propagatorOppositeToken_),
                    iSetup.getData(mkFitGeomToken_),
                    iSetup.getData(tTopoToken_),
                    tkBuilder->cloner(),
                    mkFitGeom.detLayers(),
                    mkfitSeeds.seeds(),
                    beamspot,
                    navSchool,
                    *measurementTracker,
                    *trks,
                    seedIndices,
                    hitsVecs);

  int i = 0;
  for (auto& trk : *trks) {
    for (auto& h : hitsVecs[i])
      hits->push_back(h);

    reco::TrackExtra extra;

    extra.setHits(ref_rechits, hidx, trk.numberOfValidHits());
    hidx += trk.numberOfValidHits();

    extra.setSeedRef(edm::RefToBase<TrajectorySeed>(hseeds, seedIndices[i]));

    AlgebraicVector5 v = AlgebraicVector5(0, 0, 0, 0, 0);
    reco::TrackExtra::TrajParams trajParams(trk.numberOfValidHits(), LocalTrajectoryParameters(v, 1.));
    reco::TrackExtra::Chi2sFive chi2s(trk.numberOfValidHits(), 0);
    extra.setTrajParams(std::move(trajParams), std::move(chi2s));

    extras->push_back(extra);

    trk.setExtra(reco::TrackExtraRef(ref_trackextras, idx++));

    i++;
  }

  iEvent.put(std::move(trks));
  iEvent.put(std::move(extras));
  iEvent.put(std::move(hits));

  // TODO: SeedStopInfo is currently unfilled
  iEvent.emplace(putSeedStopInfoToken_, seeds.size());
}

void MkFitOutputTrackConverter::convertCandidates(const MkFitOutputWrapper& mkFitOutput,
                                                  const mkfit::EventOfHits& eventOfHits,
                                                  const MkFitClusterIndexToHit& pixelClusterIndexToHit,
                                                  const MkFitClusterIndexToHit& stripClusterIndexToHit,
                                                  const edm::View<TrajectorySeed>& seeds,
                                                  const MagneticField& mf,
                                                  const Propagator& propagatorAlong,
                                                  const Propagator& propagatorOpposite,
                                                  const MkFitGeometry& mkFitGeom,
                                                  const TrackerTopology& tTopo,
                                                  const TkClonerImpl& hitCloner,
                                                  const std::vector<const DetLayer*>& detLayers,
                                                  const mkfit::TrackVec& mkFitSeeds,
                                                  const reco::BeamSpot* bs,
                                                  const NavigationSchool& navSchool,
                                                  const MeasurementTrackerEvent& measTk,
                                                  reco::TrackCollection& trks,
                                                  std::vector<int>& seedIndices,
                                                  std::vector<edm::OwnVector<TrackingRecHit>>& hitsVecs) const {
  const auto& candidates = mkFitOutput.tracks();
  trks.reserve(candidates.size());
  seedIndices.reserve(candidates.size());
  hitsVecs.reserve(candidates.size());

  int candIndex = -1;
  for (const auto& cand : candidates) {
    ++candIndex;
    LogTrace("MkFitOutputTrackConverter") << "Candidate " << candIndex << " pT " << cand.pT() << " eta "
                                          << cand.momEta() << " phi " << cand.momPhi() << " chi2 " << cand.chi2();

    // state: check for basic quality first
    if (cand.state().invpT() > qualityMaxInvPt_ || (qualitySignPt_ && cand.state().invpT() < 0) ||
        cand.state().theta() < qualityMinTheta_ || (M_PI - cand.state().theta()) < qualityMinTheta_ ||
        cand.state().posRsq() > qualityMaxRsq_ || std::abs(cand.state().z()) > qualityMaxZ_ ||
        (cand.state().errors.At(0, 0) + cand.state().errors.At(1, 1) + cand.state().errors.At(2, 2)) >
            qualityMaxPosErrSq_) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Candidate " << candIndex << " failed state quality checks" << cand.state().parameters;
      continue;
    }

    auto state = cand.state();  // copy because have to modify
    state.convertFromCCSToGlbCurvilinear();
    const auto& param = state.parameters;
    const auto& err = state.errors;
    AlgebraicSymMatrix55 cov;
    for (int i = 0; i < 5; ++i) {
      for (int j = i; j < 5; ++j) {
        cov[i][j] = err.At(i, j);
      }
    }

    auto fts = FreeTrajectoryState(
        GlobalTrajectoryParameters(
            GlobalPoint(param[0], param[1], param[2]), GlobalVector(param[3], param[4], param[5]), state.charge, &mf),
        CurvilinearTrajectoryError(cov));
    if (!fts.curvilinearError().posDef()) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Curvilinear error not pos-def\n"
          << fts.curvilinearError().matrix() << "\ncandidate " << candIndex << "ignored";
      continue;
    }

    //Sylvester's criterion, start from the smaller submatrix size
    double det = 0;
    if ((!fts.curvilinearError().matrix().Sub<AlgebraicSymMatrix22>(0, 0).Det(det)) || det < 0) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Fail pos-def check sub2.det for candidate " << candIndex << " with fts " << fts;
      continue;
    } else if ((!fts.curvilinearError().matrix().Sub<AlgebraicSymMatrix33>(0, 0).Det(det)) || det < 0) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Fail pos-def check sub3.det for candidate " << candIndex << " with fts " << fts;
      continue;
    } else if ((!fts.curvilinearError().matrix().Sub<AlgebraicSymMatrix44>(0, 0).Det(det)) || det < 0) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Fail pos-def check sub4.det for candidate " << candIndex << " with fts " << fts;
      continue;
    } else if ((!fts.curvilinearError().matrix().Det2(det)) || det < 0) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Fail pos-def check det for candidate " << candIndex << " with fts " << fts;
      continue;
    }

    // hits
    edm::OwnVector<TrackingRecHit> recHits;
    // nTotalHits() gives sum of valid hits (nFoundHits()) and invalid/missing hits.
    const int nhits = cand.nTotalHits();
    //std::cout << candIndex << ": " << nhits << " " << cand.nFoundHits() << std::endl;
    //bool lastHitInvalid = false;
    const auto isPhase1 = mkFitGeom.isPhase1();
    for (int i = 0; i < nhits; ++i) {
      const auto& hitOnTrack = cand.getHitOnTrack(i);
      LogTrace("MkFitOutputTrackConverter") << " hit on layer " << hitOnTrack.layer << " index " << hitOnTrack.index;
      if (hitOnTrack.index < 0) {
        // See index-desc.txt file in mkFit for description of negative values
        //
        // In order to use the regular InvalidTrackingRecHit I'd need
        // a GeomDet (and "unfortunately" that is needed in
        // TrackProducer).
        //
        // I guess we could take the track state and propagate it to
        // each layer to find the actual module the track crosses, and
        // check whether it is active or not to be able to mark
        // inactive hits
        const auto* detLayer = detLayers.at(hitOnTrack.layer);
        if (detLayer == nullptr) {
          throw cms::Exception("LogicError") << "DetLayer for layer index " << hitOnTrack.layer << " is null!";
        }
        // In principle an InvalidTrackingRecHitNoDet could be
        // inserted here, but it seems that it is best to deal with
        // them in the TrackProducer.
        //lastHitInvalid = true;
      } else {
        auto const isPixel = eventOfHits[hitOnTrack.layer].is_pixel();
        auto const& hits = isPixel ? pixelClusterIndexToHit.hits() : stripClusterIndexToHit.hits();

        auto const& thit = static_cast<BaseTrackerRecHit const&>(*hits[hitOnTrack.index]);
        if (isPhase1) {
          if (thit.firstClusterRef().isPixel() || thit.detUnit()->type().isEndcap()) {
            recHits.push_back(hits[hitOnTrack.index]->clone());
          } else {
            recHits.push_back(std::make_unique<SiStripRecHit1D>(
                thit.localPosition(),
                LocalError(thit.localPositionError().xx(), 0.f, std::numeric_limits<float>::max()),
                *thit.det(),
                thit.firstClusterRef()));
          }
        } else {
          if (thit.firstClusterRef().isPixel()) {
            recHits.push_back(hits[hitOnTrack.index]->clone());
          } else if (thit.firstClusterRef().isPhase2()) {
            recHits.push_back(std::make_unique<Phase2TrackerRecHit1D>(
                thit.localPosition(),
                LocalError(thit.localPositionError().xx(), 0.f, std::numeric_limits<float>::max()),
                *thit.det(),
                thit.firstClusterRef().cluster_phase2OT()));
          }
        }
        LogTrace("MkFitOutputTrackConverter")
            << "  pos " << recHits.back().globalPosition().x() << " " << recHits.back().globalPosition().y() << " "
            << recHits.back().globalPosition().z() << " mag2 " << recHits.back().globalPosition().mag2() << " detid "
            << recHits.back().geographicalId().rawId() << " cluster " << hitOnTrack.index;
        //lastHitInvalid = false;
      }
    }

    // MkFit hits are *not* in the order of propagation, sort by 3D radius for now (as we don't have loopers)
    // TODO: Improve the sorting (extract keys? maybe even bubble sort would work well as the hits are almost in the correct order)
    recHits.sort([&tTopo, &isPhase1](const auto& a, const auto& b) {
      //const GeomDetEnumerators::SubDetector asub = a.det()->subDetector();
      //const GeomDetEnumerators::SubDetector bsub = b.det()->subDetector();
      //const auto& apos = a.globalPosition();
      //const auto& bpos = b.globalPosition();
      // For Phase-1, can rely on subdetector index
      if (isPhase1) {
        const auto asub_ph1 = a.geographicalId().subdetId();
        const auto bsub_ph1 = b.geographicalId().subdetId();
        const auto& apos_ph1 = a.globalPosition();
        const auto& bpos_ph1 = b.globalPosition();
        if (asub_ph1 != bsub_ph1) {
          // Subdetector order (BPix, FPix, TIB, TID, TOB, TEC) corresponds also the navigation
          return asub_ph1 < bsub_ph1;
        } else {
          //if (GeomDetEnumerators::isBarrel(asub)) {
          if (isPhase1Barrel(asub_ph1)) {
            return apos_ph1.perp2() < bpos_ph1.perp2();
          } else {
            return std::abs(apos_ph1.z()) < std::abs(bpos_ph1.z());
          }
        }
      }

      // For Phase-2, can not rely uniquely on subdetector index
      const GeomDetEnumerators::SubDetector asub = a.det()->subDetector();
      const GeomDetEnumerators::SubDetector bsub = b.det()->subDetector();
      const auto& apos = a.globalPosition();
      const auto& bpos = b.globalPosition();
      const auto aid = a.geographicalId().rawId();
      const auto bid = b.geographicalId().rawId();
      const auto asubid = a.geographicalId().subdetId();
      const auto bsubid = b.geographicalId().subdetId();
      if (GeomDetEnumerators::isBarrel(asub) || GeomDetEnumerators::isBarrel(bsub)) {
        // For barrel tilted modules, or in case (only) one of the two modules is barrel, use 3D position
        if ((asubid == StripSubdetector::TOB && tTopo.tobSide(aid) < 3) ||
            (bsubid == StripSubdetector::TOB && tTopo.tobSide(bid) < 3) ||
            !(GeomDetEnumerators::isBarrel(asub) && GeomDetEnumerators::isBarrel(bsub))) {
          return apos.mag2() < bpos.mag2();
        }
        // For fully barrel comparisons and no tilt, use 2D position
        else {
          return apos.perp2() < bpos.perp2();
        }
      }
      // For fully endcap comparisons, use z position
      else {
        return std::abs(apos.z()) < std::abs(bpos.z());
      }
    });

    // seed
    const auto seedIndex = cand.label();
    LogTrace("MkFitOutputTrackConverter") << " from seed " << seedIndex << " seed hits";

    // Rescale candidate error if candidate is already propagated to first layer,
    // to be consistent with TransientInitialStateEstimator::innerState used in CkfTrackCandidateMakerBase
    // Error is only rescaled for candidates propagated to first layer;
    // otherwise, candidates undergo backwardFit where error is already rescaled

    auto tsosDet = convertInnermostState(fts, recHits, propagatorAlong, propagatorOpposite);

    if (!tsosDet.first.isValid()) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Backward fit of candidate " << candIndex << " failed, ignoring the candidate";
      continue;
    }

    TrajectoryStateOnSurface tsosState = tsosDet.first;

    TSCBLBuilderNoMaterial tscblBuilder;

    TrajectoryStateClosestToBeamLine tsAtClosestApproachTrackCand =
        tscblBuilder(*tsosState.freeState(), *bs);  //as in TrackProducerAlgorithm

    if (!(tsAtClosestApproachTrackCand.isValid())) {
      edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
      continue;
    }

    auto const& stateAtPCA = tsAtClosestApproachTrackCand.trackStateAtPCA();
    auto v0 = stateAtPCA.position();
    auto p = stateAtPCA.momentum();

    float factor = 1.f;
    if (calibrate_) {
      const float abstheta = std::fabs(tsosState.globalMomentum().theta() - 1.570796);
      const float pt = tsosState.globalMomentum().perp();
      factor = ptcorr(abstheta, pt);
    }

    math::XYZPoint pos(v0.x(), v0.y(), v0.z());
    math::XYZVector mom(p.x() * factor, p.y() * factor, p.z() * factor);  //can I just multiply p??

    int ndof = -5;
    for (auto const& recHit : recHits)
      ndof += recHit.dimension();

    //converted track
    reco::Track trk(cand.chi2(),
                    ndof,
                    pos,
                    mom,
                    stateAtPCA.charge(),
                    stateAtPCA.curvilinearError(),
                    static_cast<reco::TrackBase::TrackAlgorithm>(algo_));

    trk.appendHits(recHits.begin(), recHits.end(), tTopo);

    //extra hits (taken from TrackProducerBase<T>::setSecondHitPattern)
    const auto* outerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(recHits.back().geographicalId()));
    const auto* innerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(recHits.front().geographicalId()));
    auto const& innerCompLayers =
        navSchool.compatibleLayers(*innerLayer, fts, oppositeToMomentum);  //fts only innermost hit here
    auto const& outerCompLayers =
        navSchool.compatibleLayers(*outerLayer, fts, alongMomentum);  //fts only innermost hit here

    //use negative sigma=-3.0 in order to use a more conservative definition of isInside() for Bounds classes.
    Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);  // same as defauts....

    //inner
    for (auto it : innerCompLayers) {
      if (it->basicComponents().empty())
        continue;
      auto const& detWithState = it->compatibleDets(tsosDet.first, propagatorOpposite, estimator);
      if (detWithState.empty())
        continue;
      DetId id = detWithState.front().first->geographicalId();
      MeasurementDetWithData const& measDet = measTk.idToDet(id);
      if (measDet.isActive() && !measDet.hasBadComponents(detWithState.front().second)) {
        InvalidTrackingRecHit tmpHit(*detWithState.front().first, TrackingRecHit::missing_inner);
        trk.appendHitPattern(tmpHit, tTopo);
      } else {
        InvalidTrackingRecHit tmpHit(*detWithState.front().first, TrackingRecHit::inactive_inner);
        trk.appendHitPattern(tmpHit, tTopo);
      }
    }  //loop layers

    //outer
    for (auto it : outerCompLayers) {
      if (it->basicComponents().empty())
        continue;
      //tsosDet is innermost (not good, but does it mean anyhting is fully wrong?)
      auto const& detWithState = it->compatibleDets(tsosDet.first, propagatorAlong, estimator);
      if (detWithState.empty())
        continue;
      DetId id = detWithState.front().first->geographicalId();
      MeasurementDetWithData const& measDet = measTk.idToDet(id);
      if (measDet.isActive() && !measDet.hasBadComponents(detWithState.front().second)) {
        InvalidTrackingRecHit tmpHit(*detWithState.front().first, TrackingRecHit::missing_outer);
        trk.appendHitPattern(tmpHit, tTopo);
      } else {
        InvalidTrackingRecHit tmpHit(*detWithState.front().first, TrackingRecHit::inactive_outer);
        trk.appendHitPattern(tmpHit, tTopo);
      }
    }  //loop layers

    trks.push_back(trk);

    //need to return also seed indices and hits in some way
    seedIndices.push_back(cand.label());
    hitsVecs.push_back(recHits);
  }
}

std::pair<TrajectoryStateOnSurface, const GeomDet*> MkFitOutputTrackConverter::convertInnermostState(
    const FreeTrajectoryState& fts,
    const edm::OwnVector<TrackingRecHit>& hits,
    const Propagator& propagatorAlong,
    const Propagator& propagatorOpposite) const {
  auto det = hits[0].det();
  if (det == nullptr) {
    throw cms::Exception("LogicError") << "Got nullptr from the first hit det()";
  }

  const auto& firstHitSurface = det->surface();

  auto tsosDouble = propagatorAlong.propagateWithPath(fts, firstHitSurface);
  if (!tsosDouble.first.isValid()) {
    LogDebug("MkFitOutputTrackConverter") << "Propagating to startingState along momentum failed, trying opposite next";
    tsosDouble = propagatorOpposite.propagateWithPath(fts, firstHitSurface);
  }

  return std::make_pair(tsosDouble.first, det);
}

float MkFitOutputTrackConverter::ptcorr(const float abstheta, float pt) const {
  const int N = calibBinCenter_.size();

  if (abstheta <= calibBinCenter_[0])
    return calibBinOffset_[0] / pt + calibBinCoeff_[0];  // pt_new=a+b*pt, return pt_new/pt
  if (abstheta >= calibBinCenter_[N - 1])
    return calibBinOffset_[N - 1] / pt + calibBinCoeff_[N - 1];  // pt_new=a+b*pt, return pt_new/pt

  // interpolation
  // a_interp = a_1 + (a_2-a_1)/(x_2-x_1) * (x-x1)

  for (int i = 0; i < N - 1; ++i) {
    if (abstheta >= calibBinCenter_[i] && abstheta < calibBinCenter_[i + 1]) {
      float t = (abstheta - calibBinCenter_[i]) / (calibBinCenter_[i + 1] - calibBinCenter_[i]);
      float offset_interp = calibBinOffset_[i] + t * (calibBinOffset_[i + 1] - calibBinOffset_[i]);
      float coeff_interp = calibBinCoeff_[i] + t * (calibBinCoeff_[i + 1] - calibBinCoeff_[i]);

      return offset_interp / pt + coeff_interp;  // pt_new=a+b*pt, return pt_new/pt
    }
  }

  // Should never reach here
  return 1.f;
}

DEFINE_FWK_MODULE(MkFitOutputTrackConverter);
