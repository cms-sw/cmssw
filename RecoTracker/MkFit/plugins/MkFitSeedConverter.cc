#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/trackerHitRTTI.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// mkFit includes
#include "Track.h"

class MkFitSeedConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitSeedConverter(edm::ParameterSet const& iConfig);
  ~MkFitSeedConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  mkfit::TrackVec convertSeeds(const edm::View<TrajectorySeed>& seeds,
                               const MkFitHitIndexMap& hitIndexMap,
                               const TransientTrackingRecHitBuilder& ttrhBuilder,
                               const MagneticField& mf) const;

  using SVector3 = ROOT::Math::SVector<float, 3>;
  using SMatrixSym33 = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
  using SMatrixSym66 = ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6>>;

  const edm::EDGetTokenT<MkFitHitWrapper> hitToken_;
  const edm::EDGetTokenT<edm::View<TrajectorySeed>> seedToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::EDPutTokenT<MkFitSeedWrapper> putToken_;
};

MkFitSeedConverter::MkFitSeedConverter(edm::ParameterSet const& iConfig)
    : hitToken_{consumes<MkFitHitWrapper>(iConfig.getParameter<edm::InputTag>("hits"))},
      seedToken_{consumes<edm::View<TrajectorySeed>>(iConfig.getParameter<edm::InputTag>("seeds"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      mfToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      putToken_{produces<MkFitSeedWrapper>()} {}

void MkFitSeedConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("hits", edm::InputTag("mkFitHitConverter"));
  desc.add("seeds", edm::InputTag{"initialStepSeeds"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  descriptions.addWithDefaultLabel(desc);
}

void MkFitSeedConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  iEvent.emplace(putToken_,
                 convertSeeds(iEvent.get(seedToken_),
                              iEvent.get(hitToken_).hitIndexMap(),
                              iSetup.getData(ttrhBuilderToken_),
                              iSetup.getData(mfToken_)));
}

mkfit::TrackVec MkFitSeedConverter::convertSeeds(const edm::View<TrajectorySeed>& seeds,
                                                 const MkFitHitIndexMap& hitIndexMap,
                                                 const TransientTrackingRecHitBuilder& ttrhBuilder,
                                                 const MagneticField& mf) const {
  mkfit::TrackVec ret;
  ret.reserve(seeds.size());
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

    const auto cartError = tsos.cartesianError();  // returns a temporary, so can't chain with the following line
    const auto& cov = cartError.matrix();
    SMatrixSym66 err;
    for (int i = 0; i < 6; ++i) {
      for (int j = i; j < 6; ++j) {
        err.At(i, j) = cov[i][j];
      }
    }

    mkfit::TrackState state(tsos.charge(), pos, mom, err);
    state.convertFromCartesianToCCS();
    ret.emplace_back(state, 0, seed_index, 0, nullptr);

    // Add hits
    for (auto const& recHit : hitRange) {
      if (not trackerHitRTTI::isFromDet(recHit)) {
        throw cms::Exception("Assert") << "Encountered a seed with a hit which is not trackerHitRTTI::isFromDet()";
      }
      const auto& clusterRef = static_cast<const BaseTrackerRecHit&>(recHit).firstClusterRef();
      const auto& mkFitHit = hitIndexMap.mkFitHit(clusterRef.id(), clusterRef.index());
      ret.back().addHitIdx(mkFitHit.index(), mkFitHit.layer(), 0);  // per-hit chi2 is not known
    }
    ++seed_index;
  }
  return ret;
}

DEFINE_FWK_MODULE(MkFitSeedConverter);
