#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/trackerHitRTTI.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/MkFit/interface/MkFitInputWrapper.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// mkFit includes
#include "Hit.h"
#include "Track.h"
#include "LayerNumberConverter.h"

class MkFitInputConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitInputConverter(edm::ParameterSet const& iConfig);
  ~MkFitInputConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  template <typename HitCollection>
  void convertHits(const HitCollection& hits,
                   std::vector<mkfit::HitVec>& mkFitHits,
                   MkFitHitIndexMap& hitIndexMap,
                   int& totalHits,
                   const TrackerTopology& ttopo,
                   const TransientTrackingRecHitBuilder& ttrhBuilder,
                   const mkfit::LayerNumberConverter& lnc) const;

  bool passCCC(const SiStripRecHit2D& hit, const DetId hitId) const;
  bool passCCC(const SiPixelRecHit& hit, const DetId hitId) const;

  mkfit::TrackVec convertSeeds(const edm::View<TrajectorySeed>& seeds,
                               const MkFitHitIndexMap& hitIndexMap,
                               const TransientTrackingRecHitBuilder& ttrhBuilder,
                               const MagneticField& mf) const;

  using SVector3 = ROOT::Math::SVector<float, 3>;
  using SMatrixSym33 = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
  using SMatrixSym66 = ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6>>;

  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> seedToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  edm::EDPutTokenT<MkFitInputWrapper> putToken_;
  const float minGoodStripCharge_;
};

MkFitInputConverter::MkFitInputConverter(edm::ParameterSet const& iConfig)
    : pixelRecHitToken_{consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHits"))},
      stripRphiRecHitToken_{
          consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRphiRecHits"))},
      stripStereoRecHitToken_{
          consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripStereoRecHits"))},
      seedToken_{consumes<edm::View<TrajectorySeed>>(iConfig.getParameter<edm::InputTag>("seeds"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      mfToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      putToken_{produces<MkFitInputWrapper>()},
      minGoodStripCharge_{static_cast<float>(
          iConfig.getParameter<edm::ParameterSet>("minGoodStripCharge").getParameter<double>("value"))} {}

void MkFitInputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelRecHits", edm::InputTag{"siPixelRecHits"});
  desc.add("stripRphiRecHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stripStereoRecHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  desc.add("seeds", edm::InputTag{"initialStepSeeds"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  edm::ParameterSetDescription descCCC;
  descCCC.add<double>("value");
  desc.add("minGoodStripCharge", descCCC);

  descriptions.add("mkFitInputConverterDefault", desc);
}

void MkFitInputConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  mkfit::LayerNumberConverter lnc{mkfit::TkLayout::phase1};

  // Then import hits
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);

  std::vector<mkfit::HitVec> mkFitHits(lnc.nLayers());
  MkFitHitIndexMap hitIndexMap;
  int totalHits = 0;  // I need to have a global hit index in order to have the hit remapping working?
  // Process strips first for better memory allocation pattern
  convertHits(iEvent.get(stripRphiRecHitToken_), mkFitHits, hitIndexMap, totalHits, ttopo, ttrhBuilder, lnc);
  convertHits(iEvent.get(stripStereoRecHitToken_), mkFitHits, hitIndexMap, totalHits, ttopo, ttrhBuilder, lnc);
  convertHits(iEvent.get(pixelRecHitToken_), mkFitHits, hitIndexMap, totalHits, ttopo, ttrhBuilder, lnc);

  // Then import seeds
  auto mkFitSeeds = convertSeeds(iEvent.get(seedToken_), hitIndexMap, ttrhBuilder, iSetup.getData(mfToken_));

  iEvent.emplace(putToken_, std::move(hitIndexMap), std::move(mkFitHits), std::move(mkFitSeeds), std::move(lnc));
}

bool MkFitInputConverter::passCCC(const SiStripRecHit2D& hit, const DetId hitId) const {
  return (siStripClusterTools::chargePerCM(hitId, hit.firstClusterRef().stripCluster()) > minGoodStripCharge_);
}

bool MkFitInputConverter::passCCC(const SiPixelRecHit& hit, const DetId hitId) const { return true; }

template <typename HitCollection>
void MkFitInputConverter::convertHits(const HitCollection& hits,
                                      std::vector<mkfit::HitVec>& mkFitHits,
                                      MkFitHitIndexMap& hitIndexMap,
                                      int& totalHits,
                                      const TrackerTopology& ttopo,
                                      const TransientTrackingRecHitBuilder& ttrhBuilder,
                                      const mkfit::LayerNumberConverter& lnc) const {
  if (hits.empty())
    return;
  auto isPlusSide = [&ttopo](const DetId& detid) {
    return ttopo.side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap);
  };

  {
    const DetId detid{hits.ids().back()};
    const auto ilay =
        lnc.convertLayerNumber(detid.subdetId(), ttopo.layer(detid), false, ttopo.isStereo(detid), isPlusSide(detid));
    // Do initial reserves to minimize further memory allocations
    const auto& lastClusterRef = hits.data().back().firstClusterRef();
    hitIndexMap.resizeByClusterIndex(lastClusterRef.id(), lastClusterRef.index());
    hitIndexMap.increaseLayerSize(ilay, hits.detsetSize(hits.ids().size() - 1));
  }

  for (const auto& detset : hits) {
    const DetId detid = detset.detId();
    const auto subdet = detid.subdetId();
    const auto layer = ttopo.layer(detid);
    const auto isStereo = ttopo.isStereo(detid);
    const auto ilay = lnc.convertLayerNumber(subdet, layer, false, isStereo, isPlusSide(detid));
    hitIndexMap.increaseLayerSize(ilay, detset.size());  // to minimize memory allocations

    for (const auto& hit : detset) {
      if (!passCCC(hit, detid))
        continue;

      const auto& gpos = hit.globalPosition();
      SVector3 pos(gpos.x(), gpos.y(), gpos.z());
      const auto& gerr = hit.globalPositionError();
      SMatrixSym33 err;
      err.At(0, 0) = gerr.cxx();
      err.At(1, 1) = gerr.cyy();
      err.At(2, 2) = gerr.czz();
      err.At(0, 1) = gerr.cyx();
      err.At(0, 2) = gerr.czx();
      err.At(1, 2) = gerr.czy();

      LogTrace("MkFitInputConverter") << "Adding hit detid " << detid.rawId() << " subdet " << subdet << " layer "
                                      << layer << " isStereo " << isStereo << " zplus " << isPlusSide(detid) << " ilay "
                                      << ilay;

      hitIndexMap.insert(hit.firstClusterRef().id(),
                         hit.firstClusterRef().index(),
                         MkFitHitIndexMap::MkFitHit{static_cast<int>(mkFitHits[ilay].size()), ilay},
                         &hit);
      mkFitHits[ilay].emplace_back(pos, err, totalHits);
      ++totalHits;
    }
  }
}

mkfit::TrackVec MkFitInputConverter::convertSeeds(const edm::View<TrajectorySeed>& seeds,
                                                  const MkFitHitIndexMap& hitIndexMap,
                                                  const TransientTrackingRecHitBuilder& ttrhBuilder,
                                                  const MagneticField& mf) const {
  mkfit::TrackVec ret;
  ret.reserve(seeds.size());
  int index = 0;
  for (const auto& seed : seeds) {
    const auto hitRange = seed.recHits();
    const auto lastRecHit = ttrhBuilder.build(&*(hitRange.second - 1));
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
    ret.emplace_back(state, 0, index, 0, nullptr);

    // Add hits
    for (auto iHit = hitRange.first; iHit != hitRange.second; ++iHit) {
      if (not trackerHitRTTI::isFromDet(*iHit)) {
        throw cms::Exception("Assert") << "Encountered a seed with a hit which is not trackerHitRTTI::isFromDet()";
      }
      const auto& clusterRef = static_cast<const BaseTrackerRecHit&>(*iHit).firstClusterRef();
      const auto& mkFitHit = hitIndexMap.mkFitHit(clusterRef.id(), clusterRef.index());
      ret.back().addHitIdx(mkFitHit.index(), mkFitHit.layer(), 0);  // per-hit chi2 is not known
    }
    ++index;
  }
  return ret;
}

DEFINE_FWK_MODULE(MkFitInputConverter);
