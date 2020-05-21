#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDRecHitAlgoBase.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit MTDRecHitProducer(const edm::ParameterSet& ps);
  ~MTDRecHitProducer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  const edm::EDGetTokenT<FTLUncalibratedRecHitCollection> ftlbURecHits_;  // collection of barrel digis
  const edm::EDGetTokenT<FTLUncalibratedRecHitCollection> ftleURecHits_;  // collection of endcap digis

  const std::string ftlbInstance_;  // instance name of barrel hits
  const std::string ftleInstance_;  // instance name of endcap hits

  std::unique_ptr<MTDRecHitAlgoBase> barrel_, endcap_;

  const MTDGeometry* geom_;
};

MTDRecHitProducer::MTDRecHitProducer(const edm::ParameterSet& ps)
    : ftlbURecHits_(
          consumes<FTLUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("barrelUncalibratedRecHits"))),
      ftleURecHits_(
          consumes<FTLUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("endcapUncalibratedRecHits"))),
      ftlbInstance_(ps.getParameter<std::string>("BarrelHitsName")),
      ftleInstance_(ps.getParameter<std::string>("EndcapHitsName")) {
  produces<FTLRecHitCollection>(ftlbInstance_);
  produces<FTLRecHitCollection>(ftleInstance_);

  auto sumes = consumesCollector();

  const edm::ParameterSet& barrel = ps.getParameterSet("barrel");
  const std::string& barrelAlgo = barrel.getParameter<std::string>("algoName");
  barrel_ = MTDRecHitAlgoFactory::get()->create(barrelAlgo, barrel, sumes);

  const edm::ParameterSet& endcap = ps.getParameterSet("endcap");
  const std::string& endcapAlgo = endcap.getParameter<std::string>("algoName");
  endcap_ = MTDRecHitAlgoFactory::get()->create(endcapAlgo, endcap, sumes);
}

MTDRecHitProducer::~MTDRecHitProducer() {}

void MTDRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::ESHandle<MTDGeometry> geom;
  es.get<MTDDigiGeometryRecord>().get(geom);
  geom_ = geom.product();

  // tranparently get things from event setup
  barrel_->getEventSetup(es);
  endcap_->getEventSetup(es);

  barrel_->getEvent(evt);
  endcap_->getEvent(evt);

  // prepare output
  auto barrelRechits = std::make_unique<FTLRecHitCollection>();
  auto endcapRechits = std::make_unique<FTLRecHitCollection>();

  edm::Handle<FTLUncalibratedRecHitCollection> hBarrel;
  evt.getByToken(ftlbURecHits_, hBarrel);
  barrelRechits->reserve(hBarrel->size() / 2);
  for (const auto& uhit : *hBarrel) {
    uint32_t flags = FTLRecHit::kGood;
    auto rechit = barrel_->makeRecHit(uhit, flags);
    if (flags == FTLRecHit::kGood)
      barrelRechits->push_back(rechit);
  }

  edm::Handle<FTLUncalibratedRecHitCollection> hEndcap;
  evt.getByToken(ftleURecHits_, hEndcap);
  endcapRechits->reserve(hEndcap->size() / 2);
  for (const auto& uhit : *hEndcap) {
    uint32_t flags = FTLRecHit::kGood;
    auto rechit = endcap_->makeRecHit(uhit, flags);
    if (flags == FTLRecHit::kGood)
      endcapRechits->push_back(rechit);
  }

  // put the collection of recunstructed hits in the event
  // get the orphan handles so we can make refs for the tracking rechits
  auto barrelHandle = evt.put(std::move(barrelRechits), ftlbInstance_);
  auto endcapHandle = evt.put(std::move(endcapRechits), ftleInstance_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MTDRecHitProducer);
