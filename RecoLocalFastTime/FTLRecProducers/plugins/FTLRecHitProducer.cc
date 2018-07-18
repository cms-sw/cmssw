#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/FTLRecHitAlgoBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FTLRecHitProducer : public edm::stream::EDProducer<> {
  
 public:
  explicit FTLRecHitProducer(const edm::ParameterSet& ps);
  ~FTLRecHitProducer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  
 private:
  
  const edm::EDGetTokenT<FTLUncalibratedRecHitCollection> ftlbURecHits_; // collection of barrel digis
  const edm::EDGetTokenT<FTLUncalibratedRecHitCollection> ftleURecHits_; // collection of endcap digis
  
  const std::string ftlbInstance_; // instance name of barrel hits
  const std::string ftleInstance_; // instance name of endcap hits
  
  std::unique_ptr<FTLRecHitAlgoBase> barrel_,endcap_;
};

FTLRecHitProducer::FTLRecHitProducer(const edm::ParameterSet& ps) :
  ftlbURecHits_( consumes<FTLUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("barrelUncalibratedRecHits") ) ),
  ftleURecHits_( consumes<FTLUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("endcapUncalibratedRecHits") ) ),
  ftlbInstance_( ps.getParameter<std::string>("BarrelHitsName") ),
  ftleInstance_( ps.getParameter<std::string>("EndcapHitsName") ) {
  
  produces< FTLRecHitCollection >(ftlbInstance_);
  produces< FTLRecHitCollection >(ftleInstance_);
  
  auto sumes = consumesCollector();

  const edm::ParameterSet& barrel = ps.getParameterSet("barrel");
  const std::string& barrelAlgo = barrel.getParameter<std::string>("algoName");
  barrel_.reset( FTLRecHitAlgoFactory::get()->create(barrelAlgo, barrel, sumes) );

  const edm::ParameterSet& endcap = ps.getParameterSet("endcap");
  const std::string& endcapAlgo = endcap.getParameter<std::string>("algoName");
  endcap_.reset( FTLRecHitAlgoFactory::get()->create(endcapAlgo, endcap, sumes) );
}

FTLRecHitProducer::~FTLRecHitProducer() {
}

void
FTLRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  
 
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
  evt.put(std::move(barrelRechits), ftlbInstance_);
  evt.put(std::move(endcapRechits), ftleInstance_);
}

#include "FWCore/Framework/interface/MakerMacros.h"                                                                                                            
DEFINE_FWK_MODULE( FTLRecHitProducer );
