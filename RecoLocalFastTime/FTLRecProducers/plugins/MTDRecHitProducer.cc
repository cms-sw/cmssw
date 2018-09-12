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
  
  const edm::EDGetTokenT<FTLUncalibratedRecHitCollection> ftlbURecHits_; // collection of barrel digis
  const edm::EDGetTokenT<FTLUncalibratedRecHitCollection> ftleURecHits_; // collection of endcap digis
  
  const std::string ftlbInstance_; // instance name of barrel hits
  const std::string ftleInstance_; // instance name of endcap hits

  std::unique_ptr<MTDRecHitAlgoBase> barrel_,endcap_;
  
  edm::ESWatcher<MTDDigiGeometryRecord> geomwatcher_;
  const MTDGeometry* geom_;
};

MTDRecHitProducer::MTDRecHitProducer(const edm::ParameterSet& ps) :
  ftlbURecHits_( consumes<FTLUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("barrelUncalibratedRecHits") ) ),
  ftleURecHits_( consumes<FTLUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("endcapUncalibratedRecHits") ) ),
  ftlbInstance_( ps.getParameter<std::string>("BarrelHitsName") ),
  ftleInstance_( ps.getParameter<std::string>("EndcapHitsName") )
{
  
  produces< FTLRecHitCollection >(ftlbInstance_);
  produces< FTLRecHitCollection >(ftleInstance_);
  produces< MTDTrackingDetSetVector >();
  
  auto sumes = consumesCollector();

  const edm::ParameterSet& barrel = ps.getParameterSet("barrel");
  const std::string& barrelAlgo = barrel.getParameter<std::string>("algoName");
  barrel_.reset( MTDRecHitAlgoFactory::get()->create(barrelAlgo, barrel, sumes) );
  
  const edm::ParameterSet& endcap = ps.getParameterSet("endcap");
  const std::string& endcapAlgo = endcap.getParameter<std::string>("algoName");
  endcap_.reset( MTDRecHitAlgoFactory::get()->create(endcapAlgo, endcap, sumes) );

}

MTDRecHitProducer::~MTDRecHitProducer() {
}

void
MTDRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  
  edm::ESHandle<MTDGeometry> geom;
  if( geomwatcher_.check(es) || geom_ == nullptr ) {
    es.get<MTDDigiGeometryRecord>().get(geom);
    geom_ = geom.product();
  }
 
  // tranparently get things from event setup
  barrel_->getEventSetup(es);
  endcap_->getEventSetup(es);
  
  barrel_->getEvent(evt);
  endcap_->getEvent(evt);
  
  // prepare output
  auto barrelRechits = std::make_unique<FTLRecHitCollection>();
  auto endcapRechits = std::make_unique<FTLRecHitCollection>();
  
  edm::Handle< FTLUncalibratedRecHitCollection > hBarrel;
  evt.getByToken( ftlbURecHits_, hBarrel );  
  barrelRechits->reserve(hBarrel->size()/2);
  for(const auto& uhit : *hBarrel) {
    uint32_t flags = FTLRecHit::kGood;
    auto rechit = barrel_->makeRecHit(uhit, flags);
    if( flags == FTLRecHit::kGood ) barrelRechits->push_back( std::move(rechit) );
  }

  edm::Handle< FTLUncalibratedRecHitCollection > hEndcap;
  evt.getByToken( ftleURecHits_, hEndcap );  
  endcapRechits->reserve(hEndcap->size()/2);
  for(const auto& uhit : *hEndcap) {
    uint32_t flags = FTLRecHit::kGood;
    auto rechit = endcap_->makeRecHit(uhit, flags);
    if( flags == FTLRecHit::kGood ) endcapRechits->push_back( std::move(rechit) );
  }  

  // put the collection of recunstructed hits in the event
  // get the orphan handles so we can make refs for the tracking rechits
  auto barrelHandle = evt.put(std::move(barrelRechits), ftlbInstance_);
  auto endcapHandle = evt.put(std::move(endcapRechits), ftleInstance_);
  
  auto outputhits = std::make_unique<MTDTrackingDetSetVector>();
  auto& theoutputhits = *outputhits;

  constexpr double one_over_twelve = 1./12.;
  MeasurementError simpleRect(one_over_twelve,0,one_over_twelve);

  const auto& barrelHits = *barrelHandle;
  const auto& endcapHits = *endcapHandle;

  std::set<unsigned> geoIds; 
  std::multimap<unsigned, unsigned> geoIdToIdx;

  unsigned index = 0;
  for(const auto& hit : barrelHits) {    
    BTLDetId hitId(hit.detid());
    DetId geoId = hitId.geographicalId();
    geoIdToIdx.emplace(geoId,index);
    geoIds.emplace(geoId);
    ++index;
  }

  index = 0;
  for(const auto& hit : endcapHits) {    
    ETLDetId hitId(hit.detid());
    DetId geoId = hitId.geographicalId();
    geoIdToIdx.emplace(geoId,index);
    geoIds.emplace(geoId);
    ++index;
  }
  
  for(unsigned id : geoIds) {
    auto range = geoIdToIdx.equal_range(id);
    LocalPoint lp;
    LocalError le;

    MTDDetId mtdid(id);    
    const auto& handle = (mtdid.mtdSubDetector() == MTDDetId::BTL ? barrelHandle : endcapHandle);
    const auto& hits   = (mtdid.mtdSubDetector() == MTDDetId::BTL ? barrelHits   : endcapHits);
    const auto& thedet = geom_->idToDet(id);
    if( thedet == nullptr ) {
      throw cms::Exception("MTDRecHitProducer") << "GeographicalID: " << std::hex
						<< id
						<< " is invalid!" << std::dec
						<< std::endl;
    }
    const auto& topo = thedet->topology();
    
    MTDTrackingDetSetVector::FastFiller recHitsOnDet(theoutputhits,id);
    for(auto itr = range.first; itr != range.second; ++itr) {
      const unsigned hitidx = itr->second;
      MeasurementPoint mp(hits[hitidx].row(),hits[hitidx].column());
      lp=topo.localPosition(mp);
      le=topo.localError(mp,simpleRect);
      edm::Ref<FTLRecHitCollection> ref(handle,hitidx);
      MTDTrackingRecHit hit(lp,le,*thedet,ref);
      recHitsOnDet.push_back(hit);
    }
  }

  evt.put(std::move(outputhits));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( MTDRecHitProducer );
