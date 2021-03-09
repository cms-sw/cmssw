#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Likely.h"

#include "DataFormats/Provenance/interface/ProductID.h"
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

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// mkFit includes
#include "Hit.h"
#include "LayerNumberConverter.h"
#include "mkFit/HitStructures.h"
#include "mkFit/MkStdSeqs.h"

class MkFitHitConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitHitConverter(edm::ParameterSet const& iConfig);
  ~MkFitHitConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  template <typename HitCollection>
  edm::ProductID convertHits(const HitCollection& hits,
                             mkfit::EventOfHits& mkFitEventOfHits,
                             mkfit::HitVec& mkFitHits,
                             std::vector<TrackingRecHit const*>& clusterIndexToHit,
                             std::vector<float>& clusterChargeVec,
                             const TrackerTopology& ttopo,
                             const TransientTrackingRecHitBuilder& ttrhBuilder,
                             const MkFitGeometry& mkFitGeom) const;

  float clusterCharge(const SiStripRecHit2D& hit, DetId hitId) const;
  std::nullptr_t clusterCharge(const SiPixelRecHit& hit, DetId hitId) const;

  bool passCCC(float charge) const;
  bool passCCC(std::nullptr_t) const;  //pixel

  void setDetails(mkfit::Hit& mhit, const SiPixelCluster& cluster, const int shortId, std::nullptr_t) const;
  void setDetails(mkfit::Hit& mhit, const SiStripCluster& cluster, const int shortId, float charge) const;

  using SVector3 = ROOT::Math::SVector<float, 3>;
  using SMatrixSym33 = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
  using SMatrixSym66 = ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6>>;

  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
  const float minGoodStripCharge_;
};

MkFitHitConverter::MkFitHitConverter(edm::ParameterSet const& iConfig)
    : pixelRecHitToken_{consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHits"))},
      stripRphiRecHitToken_{
          consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRphiRecHits"))},
      stripStereoRecHitToken_{
          consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripStereoRecHits"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      wrapperPutToken_{produces<MkFitHitWrapper>()},
      clusterIndexPutToken_{produces<MkFitClusterIndexToHit>()},
      minGoodStripCharge_{static_cast<float>(
          iConfig.getParameter<edm::ParameterSet>("minGoodStripCharge").getParameter<double>("value"))} {}

void MkFitHitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelRecHits", edm::InputTag{"siPixelRecHits"});
  desc.add("stripRphiRecHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stripStereoRecHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  edm::ParameterSetDescription descCCC;
  descCCC.add<double>("value");
  desc.add("minGoodStripCharge", descCCC);

  descriptions.add("mkFitHitConverterDefault", desc);
}

void MkFitHitConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Then import hits
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper{mkFitGeom.trackerInfo()};
  mkfit::StdSeq::Cmssw_LoadHits_Begin(hitWrapper.eventOfHits(), {&hitWrapper.pixelHits(), &hitWrapper.outerHits()});

  MkFitClusterIndexToHit clusterIndexToHit;

  auto convert = [&](auto& hits, auto& mkFitHits, auto& clusterIndexToHit, auto& clusterCharge) {
    return convertHits(
        hits, hitWrapper.eventOfHits(), mkFitHits, clusterIndexToHit, clusterCharge, ttopo, ttrhBuilder, mkFitGeom);
  };
  std::vector<float> dummy;
  auto pixelClusterID =
      convert(iEvent.get(pixelRecHitToken_), hitWrapper.pixelHits(), clusterIndexToHit.pixelHits(), dummy);

  edm::ProductID stripClusterID;
  const auto& stripRphiHits = iEvent.get(stripRphiRecHitToken_);
  const auto& stripStereoHits = iEvent.get(stripStereoRecHitToken_);
  if (not stripRphiHits.empty()) {
    stripClusterID =
        convert(stripRphiHits, hitWrapper.outerHits(), clusterIndexToHit.outerHits(), hitWrapper.stripClusterCharge());
  }
  if (not stripStereoHits.empty()) {
    auto stripStereoClusterID = convert(
        stripStereoHits, hitWrapper.outerHits(), clusterIndexToHit.outerHits(), hitWrapper.stripClusterCharge());
    if (stripRphiHits.empty()) {
      stripClusterID = stripStereoClusterID;
    } else if (stripClusterID != stripStereoClusterID) {
      throw cms::Exception("LogicError") << "Encountered different cluster ProductIDs for strip RPhi hits ("
                                         << stripClusterID << ") and stereo (" << stripStereoClusterID << ")";
    }
  }
  hitWrapper.setPixelClustersID(pixelClusterID);
  hitWrapper.setOuterClustersID(stripClusterID);

  mkfit::StdSeq::Cmssw_LoadHits_End(hitWrapper.eventOfHits());

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
}

float MkFitHitConverter::clusterCharge(const SiStripRecHit2D& hit, DetId hitId) const {
  return siStripClusterTools::chargePerCM(hitId, hit.firstClusterRef().stripCluster());
}
std::nullptr_t MkFitHitConverter::clusterCharge(const SiPixelRecHit& hit, DetId hitId) const { return nullptr; }

bool MkFitHitConverter::passCCC(float charge) const { return charge > minGoodStripCharge_; }

bool MkFitHitConverter::passCCC(std::nullptr_t) const { return true; }

void MkFitHitConverter::setDetails(mkfit::Hit& mhit, const SiPixelCluster& cluster, int shortId, std::nullptr_t) const {
  mhit.setupAsPixel(shortId, cluster.sizeX(), cluster.sizeY());
}

void MkFitHitConverter::setDetails(mkfit::Hit& mhit, const SiStripCluster& cluster, int shortId, float charge) const {
  mhit.setupAsStrip(shortId, charge, cluster.amplitudes().size());
}

template <typename HitCollection>
edm::ProductID MkFitHitConverter::convertHits(const HitCollection& hits,
                                              mkfit::EventOfHits& mkFitEventOfHits,
                                              mkfit::HitVec& mkFitHits,
                                              std::vector<TrackingRecHit const*>& clusterIndexToHit,
                                              std::vector<float>& clusterChargeVec,
                                              const TrackerTopology& ttopo,
                                              const TransientTrackingRecHitBuilder& ttrhBuilder,
                                              const MkFitGeometry& mkFitGeom) const {
  if (hits.empty())
    return edm::ProductID{};
  auto isPlusSide = [&ttopo](const DetId& detid) {
    return ttopo.side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap);
  };

  constexpr bool applyCCC = std::is_same_v<typename HitCollection::data_type, SiStripRecHit2D>;

  edm::ProductID clusterID;
  {
    const auto& lastClusterRef = hits.data().back().firstClusterRef();
    clusterID = lastClusterRef.id();
    if (lastClusterRef.index() >= mkFitHits.size()) {
      auto const size = lastClusterRef.index();
      mkFitHits.resize(size);
      clusterIndexToHit.resize(size, nullptr);
      if constexpr (applyCCC) {
        clusterChargeVec.resize(size, -1.f);
      }
    }
  }

  for (const auto& detset : hits) {
    const DetId detid = detset.detId();
    const auto subdet = detid.subdetId();
    const auto layer = ttopo.layer(detid);
    const auto isStereo = ttopo.isStereo(detid);
    const auto ilay =
        mkFitGeom.layerNumberConverter().convertLayerNumber(subdet, layer, false, isStereo, isPlusSide(detid));

    for (const auto& hit : detset) {
      const auto charge = clusterCharge(hit, detid);
      if (!passCCC(charge))
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

      auto clusterRef = hit.firstClusterRef();
      if UNLIKELY (clusterRef.id() != clusterID) {
        throw cms::Exception("LogicError")
            << "Input hit collection has Refs to many cluster collections. Last hit had Ref to product " << clusterID
            << ", but encountered Ref to product " << clusterRef.id() << " on detid " << detid.rawId();
      }
      const auto clusterIndex = clusterRef.index();
      LogTrace("MkFitHitConverter") << "Adding hit detid " << detid.rawId() << " subdet " << subdet << " layer "
                                    << layer << " isStereo " << isStereo << " zplus " << isPlusSide(detid) << " index "
                                    << clusterIndex << " ilay " << ilay;

      if UNLIKELY (clusterIndex >= mkFitHits.size()) {
        mkFitHits.resize(clusterIndex + 1);
        clusterIndexToHit.resize(clusterIndex + 1, nullptr);
        if constexpr (applyCCC) {
          clusterChargeVec.resize(clusterIndex + 1, -1.f);
        }
      }
      mkFitHits[clusterIndex] = mkfit::Hit(pos, err);
      clusterIndexToHit[clusterIndex] = &hit;
      if constexpr (applyCCC) {
        clusterChargeVec[clusterIndex] = charge;
      }

      const auto uniqueIdInLayer = mkFitGeom.uniqueIdInLayer(ilay, detid.rawId());
      setDetails(mkFitHits[clusterIndex], *(hit.cluster()), uniqueIdInLayer, charge);

      mkFitEventOfHits[ilay].RegisterHit(clusterIndex);
    }
  }
  return clusterID;
}

DEFINE_FWK_MODULE(MkFitHitConverter);
