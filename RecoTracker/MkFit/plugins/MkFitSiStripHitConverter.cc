#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "convertHits.h"

namespace {
  class ConvertHitTraits {
  public:
    ConvertHitTraits(float minCharge) : minGoodStripCharge_(minCharge) {}
    using Clusters = edmNew::DetSetVector<SiStripCluster>;
    using Cluster = Clusters::data_type;

    static constexpr bool applyCCC() { return true; }
    static float chargeScale(DetId id) { return siStripClusterTools::sensorThicknessInverse(id); }
    static const Cluster& cluster(const Clusters& prod, unsigned int index) { return prod.data()[index]; }
    static float clusterCharge(const Cluster& clu, float scale) { return clu.charge() * scale; }
    bool passCCC(float charge) const { return charge > minGoodStripCharge_; }
    static void setDetails(mkfit::Hit& mhit, const Cluster& clu, int shortId, float charge) {
      mhit.setupAsStrip(shortId, charge, clu.size());
    }

  private:
    const float minGoodStripCharge_;
  };
}  // namespace

class MkFitSiStripHitConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitSiStripHitConverter(edm::ParameterSet const& iConfig);
  ~MkFitSiStripHitConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  const edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
  const edm::EDPutTokenT<std::vector<int>> layerIndexPutToken_;
  const edm::EDPutTokenT<std::vector<float>> clusterChargePutToken_;
  const ConvertHitTraits convertTraits_;
};

MkFitSiStripHitConverter::MkFitSiStripHitConverter(edm::ParameterSet const& iConfig)
    : stripRphiRecHitToken_{consumes(iConfig.getParameter<edm::InputTag>("rphiHits"))},
      stripStereoRecHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stereoHits"))},
      stripClusterToken_{consumes(iConfig.getParameter<edm::InputTag>("clusters"))},
      ttrhBuilderToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes()},
      mkFitGeomToken_{esConsumes()},
      wrapperPutToken_{produces()},
      clusterIndexPutToken_{produces()},
      layerIndexPutToken_{produces()},
      clusterChargePutToken_{produces()},
      convertTraits_{static_cast<float>(
          iConfig.getParameter<edm::ParameterSet>("minGoodStripCharge").getParameter<double>("value"))} {}

void MkFitSiStripHitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("rphiHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stereoHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  desc.add("clusters", edm::InputTag{"siStripClusters"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  edm::ParameterSetDescription descCCC;
  descCCC.add<double>("value");
  desc.add("minGoodStripCharge", descCCC);

  descriptions.add("mkFitSiStripHitConverterDefault", desc);
}

void MkFitSiStripHitConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper;
  MkFitClusterIndexToHit clusterIndexToHit;
  std::vector<int> layerIndexToHit;
  std::vector<float> clusterCharge;

  edm::ProductID stripClusterID;
  const auto& stripRphiHits = iEvent.get(stripRphiRecHitToken_);
  const auto& stripStereoHits = iEvent.get(stripStereoRecHitToken_);
  const auto maxSizeGuess(stripRphiHits.dataSize() + stripStereoHits.dataSize());
  auto const& clusters = iEvent.get(stripClusterToken_);

  auto convert = [&](auto& hits) {
    return mkfit::convertHits(convertTraits_,
                              hits,
                              clusters,
                              hitWrapper.hits(),
                              clusterIndexToHit.hits(),
                              layerIndexToHit,
                              clusterCharge,
                              ttopo,
                              ttrhBuilder,
                              mkFitGeom,
                              maxSizeGuess);
  };

  if (not stripRphiHits.empty()) {
    stripClusterID = convert(stripRphiHits);
  }
  if (not stripStereoHits.empty()) {
    auto stripStereoClusterID = convert(stripStereoHits);
    if (stripRphiHits.empty()) {
      stripClusterID = stripStereoClusterID;
    } else if (stripClusterID != stripStereoClusterID) {
      throw cms::Exception("LogicError") << "Encountered different cluster ProductIDs for strip RPhi hits ("
                                         << stripClusterID << ") and stereo (" << stripStereoClusterID << ")";
    }
  }

  hitWrapper.setClustersID(stripClusterID);

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
  iEvent.emplace(layerIndexPutToken_, std::move(layerIndexToHit));
  iEvent.emplace(clusterChargePutToken_, std::move(clusterCharge));
}

DEFINE_FWK_MODULE(MkFitSiStripHitConverter);
