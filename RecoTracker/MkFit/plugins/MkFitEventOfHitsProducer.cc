#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitEventOfHits.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit includes
#include "mkFit/HitStructures.h"
#include "mkFit/MkStdSeqs.h"

class MkFitEventOfHitsProducer : public edm::global::EDProducer<> {
public:
  explicit MkFitEventOfHitsProducer(edm::ParameterSet const& iConfig);
  ~MkFitEventOfHitsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  void fill(const std::vector<const TrackingRecHit*>& hits,
            mkfit::EventOfHits& eventOfHits,
            const MkFitGeometry& mkFitGeom) const;

  const edm::EDGetTokenT<MkFitHitWrapper> pixelHitsToken_;
  const edm::EDGetTokenT<MkFitHitWrapper> stripHitsToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> stripClusterIndexToHitToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitEventOfHits> putToken_;
};

MkFitEventOfHitsProducer::MkFitEventOfHitsProducer(edm::ParameterSet const& iConfig)
    : pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      mkFitGeomToken_{esConsumes()},
      putToken_{produces<MkFitEventOfHits>()} {}

void MkFitEventOfHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("stripHits", edm::InputTag{"mkFitSiStripHits"});

  descriptions.addWithDefaultLabel(desc);
}

void MkFitEventOfHitsProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& pixelHits = iEvent.get(pixelHitsToken_);
  const auto& stripHits = iEvent.get(stripHitsToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  auto eventOfHits = std::make_unique<mkfit::EventOfHits>(mkFitGeom.trackerInfo());
  mkfit::StdSeq::Cmssw_LoadHits_Begin(*eventOfHits, {&pixelHits.hits(), &stripHits.hits()});

  fill(iEvent.get(pixelClusterIndexToHitToken_).hits(), *eventOfHits, mkFitGeom);
  fill(iEvent.get(stripClusterIndexToHitToken_).hits(), *eventOfHits, mkFitGeom);

  mkfit::StdSeq::Cmssw_LoadHits_End(*eventOfHits);

  iEvent.emplace(putToken_, std::move(eventOfHits));
}

void MkFitEventOfHitsProducer::fill(const std::vector<const TrackingRecHit*>& hits,
                                    mkfit::EventOfHits& eventOfHits,
                                    const MkFitGeometry& mkFitGeom) const {
  for (int i = 0, end = hits.size(); i < end; ++i) {
    const auto* hit = hits[i];
    if (hit != nullptr) {
      const auto ilay = mkFitGeom.mkFitLayerNumber(hit->geographicalId());
      eventOfHits[ilay].RegisterHit(i);
    }
  }
}

DEFINE_FWK_MODULE(MkFitEventOfHitsProducer);
