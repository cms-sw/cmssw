#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

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

class MkFitSiStripHitConverterFromClusters : public edm::global::EDProducer<> {
public:
  explicit MkFitSiStripHitConverterFromClusters(edm::ParameterSet const& iConfig);
  ~MkFitSiStripHitConverterFromClusters() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  const edm::EDPutTokenT<SiStripRecHit2DCollection> rphiRecHitPutToken_;
  const edm::EDPutTokenT<SiStripRecHit2DCollection> stereoRecHitPutToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
  const edm::EDPutTokenT<std::vector<int>> layerIndexPutToken_;
  const edm::EDPutTokenT<std::vector<unsigned int>> layerSizePutToken_;
  const edm::EDPutTokenT<std::vector<float>> clusterChargePutToken_;
  edm::EDPutTokenT<SiStripMatchedRecHit2DCollection> matchedRecHitPutToken_;
  edm::EDPutTokenT<SiStripRecHit2DCollection> rphiUnmatchedRecHitPutToken_;
  edm::EDPutTokenT<SiStripRecHit2DCollection> stereoUnmatchedRecHitPutToken_;
  const ConvertHitTraits convertTraits_;
  SiStripRecHitConverterAlgorithm recHitConverterAlgorithm_;
  bool doMatching;
};

MkFitSiStripHitConverterFromClusters::MkFitSiStripHitConverterFromClusters(edm::ParameterSet const& iConfig)
    : stripClusterToken_{consumes(iConfig.getParameter<edm::InputTag>("clusters"))},
      ttrhBuilderToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes()},
      mkFitGeomToken_{esConsumes()},
      wrapperPutToken_{produces()},
      rphiRecHitPutToken_{produces(iConfig.getParameter<std::string>("rphiRecHits"))},
      stereoRecHitPutToken_{produces(iConfig.getParameter<std::string>("stereoRecHits"))},
      clusterIndexPutToken_{produces()},
      layerIndexPutToken_{produces()},
      layerSizePutToken_{produces()},
      clusterChargePutToken_{produces()},
      convertTraits_{static_cast<float>(
          iConfig.getParameter<edm::ParameterSet>("minGoodStripCharge").getParameter<double>("value"))},
      recHitConverterAlgorithm_{iConfig, consumesCollector()},
      doMatching(iConfig.getParameter<bool>("doMatching")) {
  if (doMatching) {
    matchedRecHitPutToken_ = produces(iConfig.getParameter<std::string>("matchedRecHits"));
    rphiUnmatchedRecHitPutToken_ = produces(iConfig.getParameter<std::string>("rphiRecHits") + "Unmatched");
    stereoUnmatchedRecHitPutToken_ = produces(iConfig.getParameter<std::string>("stereoRecHits") + "Unmatched");
  }
}

void MkFitSiStripHitConverterFromClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("clusters", edm::InputTag{"siStripClusters"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  desc.add<std::string>("rphiRecHits", "rphiRecHit");
  desc.add<std::string>("stereoRecHits", "stereoRecHit");

  SiStripRecHitConverterAlgorithm::fillPSetDescription(desc);

  edm::ParameterSetDescription descCCC;
  descCCC.add<double>("value");
  desc.add("minGoodStripCharge", descCCC);

  descriptions.add("mkFitSiStripHitConverterFromClustersDefault", desc);
}

void MkFitSiStripHitConverterFromClusters::produce(edm::StreamID iID,
                                                   edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper;
  MkFitClusterIndexToHit clusterIndexToHit;
  std::vector<int> layerIndexToHit;
  std::vector<float> clusterCharge;
  std::vector<unsigned int> layerSize(mkFitGeom.trackerInfo().n_layers(), 0);

  mkfit::MkFitHitFiller filler{convertTraits_,
                               // outputs
                               hitWrapper.hits(),
                               clusterIndexToHit.hits(),
                               layerIndexToHit,
                               layerSize,
                               clusterCharge,
                               // conditions
                               ttopo,
                               ttrhBuilder,
                               mkFitGeom};

  SiStripRecHitConverterAlgorithm::products stripRecHits;

  auto clusterH = iEvent.getHandle(stripClusterToken_);

  auto const size = clusterH->dataSize();
  hitWrapper.hits().resize(size);
  clusterIndexToHit.hits().resize(size, nullptr);
  layerIndexToHit.resize(size, -1);
  if constexpr (ConvertHitTraits::applyCCC()) {
    clusterCharge.resize(size, -1.f);
  }

  auto localAlgo = recHitConverterAlgorithm_.initializedClone(iSetup);
  localAlgo.run(clusterH, stripRecHits, filler);

  hitWrapper.setClustersID(clusterH.id());

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.put(rphiRecHitPutToken_, std::move(stripRecHits.rphi));
  iEvent.put(stereoRecHitPutToken_, std::move(stripRecHits.stereo));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
  iEvent.emplace(layerIndexPutToken_, std::move(layerIndexToHit));
  iEvent.emplace(layerSizePutToken_, std::move(layerSize));
  iEvent.emplace(clusterChargePutToken_, std::move(clusterCharge));

  if (doMatching) {
    iEvent.put(matchedRecHitPutToken_, std::move(stripRecHits.matched));
    iEvent.put(rphiUnmatchedRecHitPutToken_, std::move(stripRecHits.rphiUnmatched));
    iEvent.put(stereoUnmatchedRecHitPutToken_, std::move(stripRecHits.stereoUnmatched));
  }
}

DEFINE_FWK_MODULE(MkFitSiStripHitConverterFromClusters);
