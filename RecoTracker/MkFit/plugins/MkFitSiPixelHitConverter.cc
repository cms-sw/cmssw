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
    static constexpr bool applyCCC() { return false; }
    static std::nullptr_t clusterCharge(const SiPixelRecHit& hit, DetId hitId) { return nullptr; }
    static bool passCCC(std::nullptr_t) { return true; }
    static void setDetails(mkfit::Hit& mhit, const SiPixelCluster& cluster, int shortId, std::nullptr_t) {
      mhit.setupAsPixel(shortId, cluster.sizeX(), cluster.sizeY());
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
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
};

MkFitSiPixelHitConverter::MkFitSiPixelHitConverter(edm::ParameterSet const& iConfig)
    : pixelRecHitToken_{consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("hits"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      wrapperPutToken_{produces<MkFitHitWrapper>()},
      clusterIndexPutToken_{produces<MkFitClusterIndexToHit>()} {}

void MkFitSiPixelHitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("hits", edm::InputTag{"siPixelRecHits"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  descriptions.addWithDefaultLabel(desc);
}

void MkFitSiPixelHitConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper;
  MkFitClusterIndexToHit clusterIndexToHit;

  std::vector<float> dummy;
  auto pixelClusterID = mkfit::convertHits(ConvertHitTraits{},
                                           iEvent.get(pixelRecHitToken_),
                                           hitWrapper.hits(),
                                           clusterIndexToHit.hits(),
                                           dummy,
                                           ttopo,
                                           ttrhBuilder,
                                           mkFitGeom);

  hitWrapper.setClustersID(pixelClusterID);

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
}

DEFINE_FWK_MODULE(MkFitSiPixelHitConverter);
