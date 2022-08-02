#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/trackerHitRTTI.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// mkFit includes
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

class MkFitSeedConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitSeedConverter(edm::ParameterSet const& iConfig);
  ~MkFitSeedConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  mkfit::TrackVec convertSeeds(const edm::View<TrajectorySeed>& seeds,
                               const TrackerTopology& ttopo,
                               const TransientTrackingRecHitBuilder& ttrhBuilder,
                               const MagneticField& mf,
                               const MkFitGeometry& mkFitGeom) const;

  using SVector3 = ROOT::Math::SVector<float, 3>;
  using SMatrixSym33 = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
  using SMatrixSym66 = ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6>>;

  const edm::EDGetTokenT<edm::View<TrajectorySeed>> seedToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::EDPutTokenT<MkFitSeedWrapper> putToken_;
  const unsigned int maxNSeeds_;
};

MkFitSeedConverter::MkFitSeedConverter(edm::ParameterSet const& iConfig)
    : seedToken_{consumes<edm::View<TrajectorySeed>>(iConfig.getParameter<edm::InputTag>("seeds"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      mfToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      putToken_{produces<MkFitSeedWrapper>()},
      maxNSeeds_{iConfig.getParameter<unsigned int>("maxNSeeds")} {}

void MkFitSeedConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("seeds", edm::InputTag{"initialStepSeeds"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});
  desc.add("maxNSeeds", 500000U);

  descriptions.addWithDefaultLabel(desc);
}

void MkFitSeedConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  iEvent.emplace(putToken_,
                 convertSeeds(iEvent.get(seedToken_),
                              iSetup.getData(ttopoToken_),
                              iSetup.getData(ttrhBuilderToken_),
                              iSetup.getData(mfToken_),
                              iSetup.getData(mkFitGeomToken_)));
}

mkfit::TrackVec MkFitSeedConverter::convertSeeds(const edm::View<TrajectorySeed>& seeds,
                                                 const TrackerTopology& ttopo,
                                                 const TransientTrackingRecHitBuilder& ttrhBuilder,
                                                 const MagneticField& mf,
                                                 const MkFitGeometry& mkFitGeom) const {
  mkfit::TrackVec ret;
  if (seeds.size() > maxNSeeds_) {
    edm::LogError("TooManySeeds") << "Exceeded maximum number of seeds! maxNSeeds=" << maxNSeeds_
                                  << " nSeed=" << seeds.size();
    return ret;
  }
  ret.reserve(seeds.size());

  auto isPlusSide = [&ttopo](const DetId& detid) {
    return ttopo.side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap);
  };

  int seed_index = 0;
  for (const auto& seed : seeds) {
    auto const& hitRange = seed.recHits();
    const auto lastRecHit = ttrhBuilder.build(&*(hitRange.end() - 1));
    const auto tsos = trajectoryStateTransform::transientState(seed.startingState(), lastRecHit->surface(), &mf);
    const auto& stateGlobal = tsos.globalParameters();
    const auto& gpos = stateGlobal.position();
    const auto& gmom = stateGlobal.momentum();
    SVector3 pos(gpos.x(), gpos.y(), gpos.z());
    SVector3 mom(gmom.x(), gmom.y(), gmom.z());

    const auto& cov = tsos.curvilinearError().matrix();
    SMatrixSym66 err;  //fill a sub-matrix, mkfit::TrackState will convert internally
    for (int i = 0; i < 5; ++i) {
      for (int j = i; j < 5; ++j) {
        err.At(i, j) = cov[i][j];
      }
    }

    mkfit::TrackState state(tsos.charge(), pos, mom, err);
    state.convertFromGlbCurvilinearToCCS();
    ret.emplace_back(state, 0, seed_index, 0, nullptr);
    LogTrace("MkFitSeedConverter") << "Inserted seed with index " << seed_index;

    // Add hits
    for (auto const& recHit : hitRange) {
      if (not trackerHitRTTI::isFromDet(recHit)) {
        throw cms::Exception("Assert") << "Encountered a seed with a hit which is not trackerHitRTTI::isFromDet()";
      }
      auto& baseTrkRecHit = static_cast<const BaseTrackerRecHit&>(recHit);
      if (!baseTrkRecHit.isMatched()) {
        const auto& clusterRef = baseTrkRecHit.firstClusterRef();
        const auto detId = recHit.geographicalId();
        const auto ilay = mkFitGeom.layerNumberConverter().convertLayerNumber(
            detId.subdetId(), ttopo.layer(detId), false, ttopo.isStereo(detId), isPlusSide(detId));
        LogTrace("MkFitSeedConverter") << " adding hit detid " << detId.rawId() << " index " << clusterRef.index()
                                       << " ilay " << ilay;
        ret.back().addHitIdx(clusterRef.index(), ilay, 0);  // per-hit chi2 is not known
      } else {
        auto& matched2D = dynamic_cast<const SiStripMatchedRecHit2D&>(recHit);
        const OmniClusterRef* const clRefs[2] = {&matched2D.monoClusterRef(), &matched2D.stereoClusterRef()};
        const DetId detIds[2] = {matched2D.monoId(), matched2D.stereoId()};
        for (int ii = 0; ii < 2; ++ii) {
          const auto& detId = detIds[ii];
          const auto ilay = mkFitGeom.layerNumberConverter().convertLayerNumber(
              detId.subdetId(), ttopo.layer(detId), false, ttopo.isStereo(detId), isPlusSide(detId));
          LogTrace("MkFitSeedConverter") << " adding matched hit detid " << detId.rawId() << " index "
                                         << clRefs[ii]->index() << " ilay " << ilay;
          ret.back().addHitIdx(clRefs[ii]->index(), ilay, 0);  // per-hit chi2 is not known
        }
      }
    }
    ++seed_index;
  }
  return ret;
}

DEFINE_FWK_MODULE(MkFitSeedConverter);
