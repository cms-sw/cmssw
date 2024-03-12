#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "convertHits.h"

namespace {
  class ConvertHitTraitsPhase2 {
  public:
    static constexpr bool applyCCC() { return false; }
    static std::nullptr_t clusterCharge(const Phase2TrackerRecHit1D& hit, DetId hitId) { return nullptr; }
    static bool passCCC(std::nullptr_t) { return true; }
    static void setDetails(mkfit::Hit& mhit, const Phase2TrackerCluster1D& cluster, int shortId, std::nullptr_t) {
      mhit.setupAsStrip(shortId, (1 << 8) - 1, cluster.size());
    }
  };
}  // namespace

class MkFitPhase2HitConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitPhase2HitConverter(edm::ParameterSet const& iConfig);
  ~MkFitPhase2HitConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> siPhase2RecHitToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
  const edm::EDPutTokenT<std::vector<float>> clusterChargePutToken_;
  const ConvertHitTraitsPhase2 convertTraits_;
};

MkFitPhase2HitConverter::MkFitPhase2HitConverter(edm::ParameterSet const& iConfig)
    : siPhase2RecHitToken_{consumes<Phase2TrackerRecHit1DCollectionNew>(
          iConfig.getParameter<edm::InputTag>("siPhase2Hits"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      wrapperPutToken_{produces<MkFitHitWrapper>()},
      clusterIndexPutToken_{produces<MkFitClusterIndexToHit>()},
      clusterChargePutToken_{produces<std::vector<float>>()},
      convertTraits_{} {}

void MkFitPhase2HitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("siPhase2Hits", edm::InputTag{"siPhase2RecHits"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  descriptions.add("mkFitPhase2HitConverterDefault", desc);
}

void MkFitPhase2HitConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper;
  MkFitClusterIndexToHit clusterIndexToHit;
  std::vector<float> clusterCharge;

  edm::ProductID stripClusterID;
  const auto& phase2Hits = iEvent.get(siPhase2RecHitToken_);
  std::vector<float> dummy;
  if (not phase2Hits.empty()) {
    stripClusterID = mkfit::convertHits(ConvertHitTraitsPhase2{},
                                        phase2Hits,
                                        hitWrapper.hits(),
                                        clusterIndexToHit.hits(),
                                        dummy,
                                        ttopo,
                                        ttrhBuilder,
                                        mkFitGeom);
  }

  hitWrapper.setClustersID(stripClusterID);

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
  iEvent.emplace(clusterChargePutToken_, std::move(clusterCharge));
}

DEFINE_FWK_MODULE(MkFitPhase2HitConverter);
