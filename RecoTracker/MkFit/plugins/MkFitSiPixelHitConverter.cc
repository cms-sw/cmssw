#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "convertHits.h"

namespace {
  struct ConvertHitTraits {
    using Clusters = SiPixelClusterCollectionNew;
    using Cluster = Clusters::data_type;

    static constexpr bool applyCCC() { return false; }
    static float chargeScale(DetId) { return 0; }
    static const Cluster& cluster(const Clusters& prod, unsigned int index) { return prod.data()[index]; }
    static std::nullptr_t clusterCharge(const Cluster&, float) { return nullptr; }
    static bool passCCC(std::nullptr_t) { return true; }
    static void setDetails(mkfit::Hit& mhit, const Cluster& clu, int shortId, std::nullptr_t) {
      mhit.setupAsPixel(shortId, clu.sizeX(), clu.sizeY());
    }
  };
}  // namespace

class MkFitSiPixelHitConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitSiPixelHitConverter(edm::ParameterSet const& iConfig);
  ~MkFitSiPixelHitConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> pixelClusterToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
  const edm::EDPutTokenT<std::vector<int>> layerIndexPutToken_;
};

MkFitSiPixelHitConverter::MkFitSiPixelHitConverter(edm::ParameterSet const& iConfig)
    : pixelRecHitToken_{consumes(iConfig.getParameter<edm::InputTag>("hits"))},
      pixelClusterToken_{consumes(iConfig.getParameter<edm::InputTag>("clusters"))},
      ttrhBuilderToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes()},
      mkFitGeomToken_{esConsumes()},
      wrapperPutToken_{produces()},
      clusterIndexPutToken_{produces()},
      layerIndexPutToken_{produces()} {}

void MkFitSiPixelHitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("hits", edm::InputTag{"siPixelRecHits"});
  desc.add("clusters", edm::InputTag{"siPixelClusters"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  descriptions.addWithDefaultLabel(desc);
}

void MkFitSiPixelHitConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper;
  MkFitClusterIndexToHit clusterIndexToHit;
  std::vector<int> layerIndexToHit;

  std::vector<float> dummy;
  auto const& hits = iEvent.get(pixelRecHitToken_);
  auto pixelClusterID = mkfit::convertHits(ConvertHitTraits{},
                                           hits,
                                           iEvent.get(pixelClusterToken_),
                                           hitWrapper.hits(),
                                           clusterIndexToHit.hits(),
                                           layerIndexToHit,
                                           dummy,
                                           ttopo,
                                           ttrhBuilder,
                                           mkFitGeom,
                                           hits.dataSize());

  hitWrapper.setClustersID(pixelClusterID);

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
  iEvent.emplace(layerIndexPutToken_, std::move(layerIndexToHit));
}

DEFINE_FWK_MODULE(MkFitSiPixelHitConverter);
