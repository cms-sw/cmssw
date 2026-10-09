#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
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
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
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
#include "RecoTracker/MkFitCore/interface/HitStateOnTrack.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>

namespace {
  // a final-fit state on a hit's module (mkfit::HitStateOnTrack: that module's local frame)
  TrajectoryStateOnSurface toTSOS(const mkfit::HitStateOnTrack& st, const GeomDet& det, const MagneticField& mf) {
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j)
        m(i, j) = st.err[i * (i + 1) / 2 + j];
    return TrajectoryStateOnSurface(
        LocalTrajectoryParameters(st.par[0], st.par[1], st.par[2], st.par[3], st.par[4], st.pzSign, true),
        LocalTrajectoryError(m),
        det.surface(),
        &mf);
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
                         std::vector<edm::OwnVector<TrackingRecHit>>& hitsVecs,
                         std::vector<int>& candIndices,
                         std::vector<std::vector<int>>& hotVecs) const;

  // validation of the per-hit states (validateHitStates): the mkFit smoother against TrajectoryStateCombiner
  void validateHitStates(const MkFitOutputWrapper& mkFitOutput,
                         const std::vector<int>& candIndices,
                         const std::vector<edm::OwnVector<TrackingRecHit>>& hitsVecs,
                         const std::vector<std::vector<int>>& hotVecs,
                         const MagneticField& mf) const;

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

  const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventToken_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navToken_;

  const int algo_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

  // trajectoryInEvent: also produce the Trajectories and their association to the tracks (as TrackProducer); needs
  // the final fit's per-hit states (MkFitFitProducer storeHitStates).  With per-hit states the TrackExtras are filled
  // in full either way.
  const bool trajectoryInEvent_;
  const bool validateHitStates_;
  edm::EDPutTokenT<std::vector<Trajectory>> putTrajToken_;
  edm::EDPutTokenT<TrajTrackAssociationCollection> putTrajAssocToken_;
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
      measurementTrackerEventToken_{consumes(iConfig.getParameter<edm::InputTag>("measurementTrackerEvent"))},
      navToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("NavigationSchool"))},
      algo_{reco::TrackBase::algoByName(
          iConfig.getParameter<std::string>("algorithmName").empty()
              ? TString(iConfig.getParameter<edm::InputTag>("seeds").label()).ReplaceAll("Seeds", "").Data()
              : iConfig.getParameter<std::string>("algorithmName"))},
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      trajectoryInEvent_{iConfig.getParameter<bool>("trajectoryInEvent")},
      validateHitStates_{iConfig.getUntrackedParameter<bool>("validateHitStates")} {
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  if (trajectoryInEvent_) {
    putTrajToken_ = produces<std::vector<Trajectory>>();
    putTrajAssocToken_ = produces<TrajTrackAssociationCollection>();
  }
}

void MkFitOutputTrackConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("mkFitEventOfHits", edm::InputTag{"mkFitEventOfHits"});
  desc.add("mkFitPixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("mkFitStripHits", edm::InputTag{"mkFitSiStripHits"});
  desc.add("mkFitSeeds", edm::InputTag{"mkFitSeedConverter"});
  desc.add("src", edm::InputTag{"mkFitProducer"});
  desc.add("seeds", edm::InputTag{"initialStepSeeds"});
  desc.add<std::string>("algorithmName", "")->setComment("track algorithm; if empty, the seeds label without 'Seeds'");
  desc.add("beamSpot", edm::InputTag{"offlineBeamSpot"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});
  desc.add("propagatorAlong", edm::ESInputTag{"", "PropagatorWithMaterial"});
  desc.add("propagatorOpposite", edm::ESInputTag{"", "PropagatorWithMaterialOpposite"});

  desc.add<double>("qualityMaxInvPt", 100)->setComment("max(1/pt) for converted tracks");
  desc.add<double>("qualityMinTheta", 0.01)->setComment("lower bound on theta (or pi-theta) for converted tracks");
  desc.add<double>("qualityMaxR", 120)->setComment("max(R) for the state position for converted tracks");
  desc.add<double>("qualityMaxZ", 280)->setComment("max(|Z|) for the state position for converted tracks");
  desc.add<double>("qualityMaxPosErr", 100)->setComment("max position error for converted tracks");
  desc.add<bool>("qualitySignPt", true)->setComment("check sign of 1/pt for converted tracks");

  desc.add<edm::ESInputTag>("NavigationSchool", edm::ESInputTag{"", "SimpleNavigationSchool"});
  desc.add<bool>("trajectoryInEvent", false)
      ->setComment("also produce Trajectories and the trajectory-track association; needs per-hit states (src)");
  desc.addUntracked<bool>("validateHitStates", false)
      ->setComment("compare the per-hit smoothed states with TrajectoryStateCombiner (validation only)");
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

  std::unique_ptr<reco::TrackCollection> trks = std::make_unique<reco::TrackCollection>();
  std::unique_ptr<TrackingRecHitCollection> hits = std::make_unique<TrackingRecHitCollection>();
  std::unique_ptr<reco::TrackExtraCollection> extras = std::make_unique<reco::TrackExtraCollection>();

  std::vector<int> seedIndices;
  std::vector<edm::OwnVector<TrackingRecHit>> hitsVecs;
  std::vector<int> candIndices;
  std::vector<std::vector<int>> hotVecs;

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
                    hitsVecs,
                    candIndices,
                    hotVecs);

  // per-hit states of the final fit, aligned with the tracks of src (empty: not stored, or no tracks in the event)
  const auto& mkFitOutput = iEvent.get(tracksToken_);
  const auto& hitStates = mkFitOutput.hitStates();
  const bool haveStates = !hitStates.empty();
  if (trajectoryInEvent_ && !haveStates && !mkFitOutput.tracks().empty())
    throw cms::Exception("Configuration") << "MkFitOutputTrackConverter: trajectoryInEvent needs the final fit's "
                                             "per-hit states (MkFitFitProducer storeHitStates = True)";
  const auto& mf = iSetup.getData(mfToken_);
  const auto& detLayers = mkFitGeom.detLayers();
  auto trajs = std::make_unique<std::vector<Trajectory>>();
  if (trajectoryInEvent_)
    trajs->reserve(trks->size());

  int i = 0;
  for (auto& trk : *trks) {
    for (auto& h : hitsVecs[i])
      hits->push_back(h);

    if (haveStates) {
      // as TrackProducer (KfTrackProducerBase::putInEvt): inner and outer state, a local state and a chi2 per hit
      const auto& hs = hitStates[candIndices[i]];
      const auto& rh = hitsVecs[i];
      const auto& hot = hotVecs[i];
      const unsigned int nh = rh.size();
      const auto innerTsos = toTSOS(hs[hot.front()], *rh.front().det(), mf);
      const auto outerTsos = toTSOS(hs[hot.back()], *rh.back().det(), mf);
      const auto& ip = innerTsos.globalPosition();
      const auto& im = innerTsos.globalMomentum();
      const auto& op = outerTsos.globalPosition();
      const auto& om = outerTsos.globalMomentum();
      reco::TrackExtra extra(math::XYZPoint(op.x(), op.y(), op.z()),
                             math::XYZVector(om.x(), om.y(), om.z()),
                             true,
                             math::XYZPoint(ip.x(), ip.y(), ip.z()),
                             math::XYZVector(im.x(), im.y(), im.z()),
                             true,
                             outerTsos.curvilinearError().matrix(),
                             rh.back().geographicalId().rawId(),
                             innerTsos.curvilinearError().matrix(),
                             rh.front().geographicalId().rawId(),
                             alongMomentum,
                             edm::RefToBase<TrajectorySeed>(hseeds, seedIndices[i]));
      extra.setHits(ref_rechits, hidx, nh);
      hidx += nh;
      reco::TrackExtra::TrajParams trajParams;
      reco::TrackExtra::Chi2sFive chi2s;
      trajParams.reserve(nh);
      chi2s.reserve(nh);
      for (unsigned int k = 0; k < nh; ++k) {
        const auto& st = hs[hot[k]];
        trajParams.emplace_back(st.par[0], st.par[1], st.par[2], st.par[3], st.par[4], st.pzSign, true);
        chi2s.push_back(Traj2TrackHits::toChi2x5(st.chi2));
      }
      extra.setTrajParams(std::move(trajParams), std::move(chi2s));
      extras->push_back(extra);

      if (trajectoryInEvent_) {
        Trajectory traj(std::shared_ptr<const TrajectorySeed>(&seeds[seedIndices[i]], edm::do_nothing_deleter()),
                        alongMomentum);
        traj.setSeedRef(edm::RefToBase<TrajectorySeed>(hseeds, seedIndices[i]));
        traj.reserve(nh);
        for (unsigned int k = 0; k < nh; ++k) {
          const auto& st = hs[hot[k]];
          const auto* det = rh[k].det();
          traj.push(TrajectoryMeasurement(k == 0        ? innerTsos
                                          : k == nh - 1 ? outerTsos
                                                        : toTSOS(st, *det, mf),
                                          rh[k].cloneSH(),
                                          st.chi2,
                                          detLayers.at(mkFitGeom.mkFitLayerNumber(rh[k].geographicalId()))),
                    st.chi2);
        }
        trajs->push_back(std::move(traj));
      }
    } else {
      reco::TrackExtra extra;

      extra.setHits(ref_rechits, hidx, trk.numberOfValidHits());
      hidx += trk.numberOfValidHits();

      extra.setSeedRef(edm::RefToBase<TrajectorySeed>(hseeds, seedIndices[i]));

      AlgebraicVector5 v = AlgebraicVector5(0, 0, 0, 0, 0);
      reco::TrackExtra::TrajParams trajParams(trk.numberOfValidHits(), LocalTrajectoryParameters(v, 1.));
      reco::TrackExtra::Chi2sFive chi2s(trk.numberOfValidHits(), 0);
      extra.setTrajParams(std::move(trajParams), std::move(chi2s));

      extras->push_back(extra);
    }

    trk.setExtra(reco::TrackExtraRef(ref_trackextras, idx++));

    i++;
  }

  if (validateHitStates_ && haveStates && !mkFitOutput.hitStatesFwd().empty())
    validateHitStates(mkFitOutput, candIndices, hitsVecs, hotVecs, mf);

  auto rTracks = iEvent.put(std::move(trks));
  iEvent.put(std::move(extras));
  iEvent.put(std::move(hits));
  if (trajectoryInEvent_) {
    auto rTrajs = iEvent.emplace(putTrajToken_, std::move(*trajs));
    TrajTrackAssociationCollection assoc(rTrajs, rTracks);
    for (unsigned int k = 0; k < rTrajs->size(); ++k)
      assoc.insert(edm::Ref<std::vector<Trajectory>>(rTrajs, k), reco::TrackRef(rTracks, k));
    iEvent.emplace(putTrajAssocToken_, std::move(assoc));
  }

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
                                                  std::vector<edm::OwnVector<TrackingRecHit>>& hitsVecs,
                                                  std::vector<int>& candIndices,
                                                  std::vector<std::vector<int>>& hotVecs) const {
  const auto& candidates = mkFitOutput.tracks();
  trks.reserve(candidates.size());
  seedIndices.reserve(candidates.size());
  hitsVecs.reserve(candidates.size());

  int candIndex = -1;
  for (const auto& cand : candidates) {
    ++candIndex;
    LogTrace("MkFitOutputTrackConverter") << "Candidate " << candIndex << " pT " << cand.pT() << " eta "
                                          << cand.momEta() << " phi " << cand.momPhi() << " chi2 " << cand.chi2();

    // state and chi2: check for basic quality first (a fit with a non-finite or negative chi2 has failed, as
    // KFFittingSmoother treats it)
    if (cand.state().invpT() > qualityMaxInvPt_ || (qualitySignPt_ && cand.state().invpT() < 0) ||
        cand.state().theta() < qualityMinTheta_ || (M_PI - cand.state().theta()) < qualityMinTheta_ ||
        cand.state().posRsq() > qualityMaxRsq_ || std::abs(cand.state().z()) > qualityMaxZ_ ||
        (cand.state().errors.At(0, 0) + cand.state().errors.At(1, 1) + cand.state().errors.At(2, 2)) >
            qualityMaxPosErrSq_ ||
        edm::isNotFinite(cand.chi2()) || cand.chi2() < 0.f) {
      edm::LogInfo("MkFitOutputTrackConverter") << "Candidate " << candIndex << " failed state quality checks"
                                                << cand.state().parameters << " chi2 " << cand.chi2();
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
    std::vector<std::pair<const TrackingRecHit*, int>> hitHoT;  // rec hit -> HitOnTrack position
    std::vector<std::pair<float, float>> hitR2Z;                // mkFit hit R^2 and z, as hitHoT
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
        hitHoT.emplace_back(&recHits.back(), i);
        const auto* mkHitPos = eventOfHits[hitOnTrack.layer].refHit(hitOnTrack.index).posArray();
        hitR2Z.emplace_back(mkHitPos[0] * mkHitPos[0] + mkHitPos[1] * mkHitPos[1], mkHitPos[2]);
        LogTrace("MkFitOutputTrackConverter")
            << "  pos " << recHits.back().globalPosition().x() << " " << recHits.back().globalPosition().y() << " "
            << recHits.back().globalPosition().z() << " mag2 " << recHits.back().globalPosition().mag2() << " detid "
            << recHits.back().geographicalId().rawId() << " cluster " << hitOnTrack.index;
        //lastHitInvalid = false;
      }
    }

    // The hits in the order of the final fit (MkFitter::reFitIndices): by R^2 + (z - z0)^2 of the mkFit hit positions,
    // z0 the z of the hit with the smallest R, ties by HitOnTrack position.  recHits[0] is then the hit on whose plane
    // the fit's innermost state lies.
    std::unordered_map<const TrackingRecHit*, std::pair<float, int>> fitOrder;
    {
      float minR2 = std::numeric_limits<float>::max(), z0 = 0.f;
      for (const auto& [r2, z] : hitR2Z)
        if (r2 < minR2) {
          minR2 = r2;
          z0 = z;
        }
      for (unsigned int k = 0; k < hitHoT.size(); ++k) {
        const float dz = hitR2Z[k].second - z0;
        fitOrder.emplace(hitHoT[k].first, std::make_pair(hitR2Z[k].first + dz * dz, hitHoT[k].second));
      }
    }
    recHits.sort([&fitOrder](const auto& a, const auto& b) { return fitOrder.at(&a) < fitOrder.at(&b); });

    // seed
    const auto seedIndex = cand.label();
    LogTrace("MkFitOutputTrackConverter") << " from seed " << seedIndex << " seed hits";

    // Rescale candidate error if candidate is already propagated to first layer,
    // to be consistent with TransientInitialStateEstimator::innerState used in CkfTrackCandidateMakerBase
    // Error is only rescaled for candidates propagated to first layer;
    // otherwise, candidates undergo backwardFit where error is already rescaled

    // The backward refit leaves the state on the plane of the hit that reFitIndices
    // ordered innermost (smallest R), so no propagation is needed here -- propagating
    // would re-apply that module material a second time.  recHits was sorted above by
    // 3D distance from the origin; for a track from the beam line the two agree.
    auto detH0 = recHits[0].det();

    if (detH0 == nullptr) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Got nullptr from the first hit det() " << candIndex << " failed, ignoring the candidate";
      continue;
    }

    auto tsosState = TrajectoryStateOnSurface(fts, detH0->surface());

    if (!tsosState.isValid()) {
      edm::LogInfo("MkFitOutputTrackConverter")
          << "Backward fit of candidate " << candIndex << " failed, ignoring the candidate";
      continue;
    }

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

    math::XYZPoint pos(v0.x(), v0.y(), v0.z());
    math::XYZVector mom(p.x(), p.y(), p.z());

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

    // the HitOnTrack position of every (sorted) rec hit; with per-hit states, each must have a valid one
    std::vector<int> hot;
    hot.reserve(recHits.size());
    for (const auto& h : recHits)
      for (const auto& [ptr, pos] : hitHoT)
        if (ptr == &h) {
          hot.push_back(pos);
          break;
        }
    if (!mkFitOutput.hitStates().empty()) {
      const auto& hs = mkFitOutput.hitStates()[candIndex];
      bool ok = hot.size() == recHits.size();
      for (int pos : hot)
        ok = ok && pos < (int)hs.size() && hs[pos].valid;
      if (!ok) {
        edm::LogInfo("MkFitOutputTrackConverter") << "Candidate " << candIndex << " has no valid final-fit state at "
                                                  << "every hit, ignoring the candidate";
        continue;
      }
    }
    // the state for the navigation outward: the final fit's state at its outermost hit when that is the outermost
    // rec hit (the same with and without per-hit states), else the innermost one
    const auto& outer = mkFitOutput.outerStates();
    const bool useOuter =
        !outer.empty() && !hot.empty() && outer[candIndex].hot == hot.back() && outer[candIndex].state.valid;
    const auto outerTsos = useOuter ? toTSOS(outer[candIndex].state, *recHits.back().det(), mf) : tsosState;

    //extra hits (taken from TrackProducerBase<T>::setSecondHitPattern)
    const auto* outerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(recHits.back().geographicalId()));
    const auto* innerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(recHits.front().geographicalId()));
    auto const& innerCompLayers =
        navSchool.compatibleLayers(*innerLayer, fts, oppositeToMomentum);  //fts only innermost hit here
    auto const& outerCompLayers = navSchool.compatibleLayers(*outerLayer, *outerTsos.freeState(), alongMomentum);

    //use negative sigma=-3.0 in order to use a more conservative definition of isInside() for Bounds classes.
    Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);  // same as defauts....

    // Lost hits between the innermost and the outermost hit (beyond them, the navigation below finds the missing
    // ones): the hits the final fit removed as outliers, recorded on their own modules as the CKF fit records its
    // outliers. They follow the valid hits, so the track hits of the hit pattern stay in step with the rec hits.
    {
      const auto [firstPos, lastPos] = std::minmax_element(hot.begin(), hot.end());
      for (int pos = *firstPos + 1; pos < *lastPos; ++pos) {
        const auto& lostHoT = cand.getHitOnTrack(pos);
        if (!mkfit::Hit::isFitOutlierIdx(lostHoT.index))
          continue;
        auto const& hits =
            eventOfHits[lostHoT.layer].is_pixel() ? pixelClusterIndexToHit.hits() : stripClusterIndexToHit.hits();
        InvalidTrackingRecHit lostHit(*hits[mkfit::Hit::hitIdxOfFitOutlier(lostHoT.index)]->det(),
                                      TrackingRecHit::missing);
        trk.appendHitPattern(lostHit, tTopo);
      }
    }

    //inner
    for (auto it : innerCompLayers) {
      if (it->basicComponents().empty())
        continue;
      auto const& detWithState = it->compatibleDets(tsosState, propagatorOpposite, estimator);
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
      auto const& detWithState = it->compatibleDets(outerTsos, propagatorAlong, estimator);
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
    candIndices.push_back(candIndex);
    hotVecs.push_back(std::move(hot));
  }
}

void MkFitOutputTrackConverter::validateHitStates(const MkFitOutputWrapper& mkFitOutput,
                                                  const std::vector<int>& candIndices,
                                                  const std::vector<edm::OwnVector<TrackingRecHit>>& hitsVecs,
                                                  const std::vector<std::vector<int>>& hotVecs,
                                                  const MagneticField& mf) const {
  // mkFit's single-precision smoothed state against the double-precision combination of the same two states
  TrajectoryStateCombiner combiner;
  int n = 0, nFail = 0;
  double maxPull[5] = {0}, sumPull[5] = {0}, maxErrRel[5] = {0};
  for (unsigned int t = 0; t < candIndices.size(); ++t) {
    const auto& sm = mkFitOutput.hitStates()[candIndices[t]];
    const auto& fw = mkFitOutput.hitStatesFwd()[candIndices[t]];
    const auto& bw = mkFitOutput.hitStatesBwd()[candIndices[t]];
    for (unsigned int k = 0; k < hotVecs[t].size(); ++k) {
      const int pos = hotVecs[t][k];
      if (sm[pos].kind != mkfit::HitStateOnTrack::Combined)
        continue;
      if (!fw[pos].valid || !bw[pos].valid) {
        ++nFail;
        continue;
      }
      const auto* det = hitsVecs[t][k].det();
      const auto c = combiner(toTSOS(fw[pos], *det, mf), toTSOS(bw[pos], *det, mf));
      if (!c.isValid()) {
        ++nFail;
        continue;
      }
      const auto& cp = c.localParameters().vector();
      const auto& ce = c.localError().matrix();
      ++n;
      for (int j = 0; j < 5; ++j) {
        const double sig = std::sqrt(ce(j, j));
        const double pull = std::abs(sm[pos].par[j] - cp[j]) / sig;
        const double rel = std::abs(std::sqrt(sm[pos].err[j * (j + 3) / 2]) / sig - 1.);
        maxPull[j] = std::max(maxPull[j], pull);
        sumPull[j] += pull;
        maxErrRel[j] = std::max(maxErrRel[j], rel);
      }
    }
  }
  edm::LogPrint log("MkFitHitStateValidation");
  log << "combined hits " << n << " failed " << nFail << " | |d par|/sigma mean/max (q/p, dxdz, dydz, x, y):";
  for (int j = 0; j < 5; ++j)
    log << " " << (n ? sumPull[j] / n : 0.) << "/" << maxPull[j];
  log << " | max |sigma ratio - 1|:";
  for (int j = 0; j < 5; ++j)
    log << " " << maxErrRel[j];
}

DEFINE_FWK_MODULE(MkFitOutputTrackConverter);
