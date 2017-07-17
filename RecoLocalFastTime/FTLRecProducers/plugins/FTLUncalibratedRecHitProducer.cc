#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/FTLUncalibratedRecHitAlgoBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FTLUncalibratedRecHitProducer : public edm::stream::EDProducer<> {
  
 public:
  explicit FTLUncalibratedRecHitProducer(const edm::ParameterSet& ps);
  ~FTLUncalibratedRecHitProducer();
  virtual void produce(edm::Event& evt, const edm::EventSetup& es) override;
  
 private:
  
  const edm::EDGetTokenT<FTLDigiCollection> ftlbDigis_; // collection of HGCEE digis
  const edm::EDGetTokenT<FTLDigiCollection> ftleDigis_; // collection of HGCHEF digis
  
  const std::string ftlbInstance_; // instance name of barrel hits
  const std::string ftleInstance_; // instance name of endcap hits
  
  std::unique_ptr<FTLUncalibratedRecHitAlgoBase> barrel_,endcap_;
};

FTLUncalibratedRecHitProducer::FTLUncalibratedRecHitProducer(const edm::ParameterSet& ps) :
  ftlbDigis_( consumes<FTLDigiCollection>( ps.getParameter<edm::InputTag>("barrelDigis") ) ),
  ftleDigis_( consumes<FTLDigiCollection>( ps.getParameter<edm::InputTag>("endcapDigis") ) ),
  ftlbInstance_( ps.getParameter<std::string>("BarrelHitsName") ),
  ftleInstance_( ps.getParameter<std::string>("EndcapHitsName") ) {
  
  produces< FTLUncalibratedRecHitCollection >(ftlbInstance_);
  produces< FTLUncalibratedRecHitCollection >(ftleInstance_);
  
  auto sumes = consumesCollector();

  const edm::ParameterSet& barrel = ps.getParameterSet("barrel");
  const std::string& barrelAlgo = barrel.getParameter<std::string>("algoName");
  barrel_.reset( FTLUncalibratedRecHitAlgoFactory::get()->create(barrelAlgo, barrel, sumes) );

  const edm::ParameterSet& endcap = ps.getParameterSet("endcap");
  const std::string& endcapAlgo = endcap.getParameter<std::string>("algoName");
  endcap_.reset( FTLUncalibratedRecHitAlgoFactory::get()->create(endcapAlgo, endcap, sumes) );
}

FTLUncalibratedRecHitProducer::~FTLUncalibratedRecHitProducer() {
}

void
FTLUncalibratedRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  
 
  // tranparently get things from event setup
  barrel_->getEventSetup(es);
  endcap_->getEventSetup(es);

  barrel_->getEvent(evt);
  endcap_->getEvent(evt);
  
  // prepare output
  auto barrelRechits = std::make_unique<FTLUncalibratedRecHitCollection>();
  auto endcapRechits = std::make_unique<FTLUncalibratedRecHitCollection>();
  
  edm::Handle< FTLDigiCollection > hBarrel;
  evt.getByToken( ftlbDigis_, hBarrel );  
  barrelRechits->reserve(hBarrel->size()/2);
  for(const auto& digi : *hBarrel) {
    barrelRechits->push_back( std::move(barrel_->makeRecHit(digi)) );
  }

  edm::Handle< FTLDigiCollection > hEndcap;
  evt.getByToken( ftleDigis_, hEndcap );  
  endcapRechits->reserve(hEndcap->size()/2);
  for(const auto& digi : *hEndcap) {
    endcapRechits->push_back( std::move(endcap_->makeRecHit(digi)) );
  }
      
  // put the collection of recunstructed hits in the event
  evt.put(std::move(barrelRechits), ftlbInstance_);
  evt.put(std::move(endcapRechits), ftleInstance_);
}

#include "FWCore/Framework/interface/MakerMacros.h"                                                                                                            
DEFINE_FWK_MODULE( FTLUncalibratedRecHitProducer );
