/** \class HGCalRecHitProducer
 *   produce HGCAL rechits from uncalibrated rechits
 *
 *  simplified version of Ecal code
 *
 *  \author Valeri Andreev (ported to 76X by L. Gray)
 *
 **/
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerBaseClass.h"

class HGCalRecHitProducer : public edm::stream::EDProducer<> {
  
 public:
  explicit HGCalRecHitProducer(const edm::ParameterSet& ps);
  ~HGCalRecHitProducer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  
 private:
  
  const edm::EDGetTokenT<HGCeeUncalibratedRecHitCollection> eeUncalibRecHitCollection_;
  const edm::EDGetTokenT<HGChefUncalibratedRecHitCollection>  hefUncalibRecHitCollection_;
  const edm::EDGetTokenT<HGChebUncalibratedRecHitCollection> hebUncalibRecHitCollection_;
  const std::string eeRechitCollection_; // instance name for HGCEE
  const std::string hefRechitCollection_; // instance name for HGCHEF
  const std::string hebRechitCollection_; // instance name for HGCHEB 
  
  std::unique_ptr<HGCalRecHitWorkerBaseClass> worker_;  
};

HGCalRecHitProducer::HGCalRecHitProducer(const edm::ParameterSet& ps) :
  eeUncalibRecHitCollection_( consumes<HGCeeUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("HGCEEuncalibRecHitCollection") ) ),
  hefUncalibRecHitCollection_( consumes<HGChefUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("HGCHEFuncalibRecHitCollection") ) ),
  hebUncalibRecHitCollection_( consumes<HGChebUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("HGCHEBuncalibRecHitCollection") ) ),
  eeRechitCollection_( ps.getParameter<std::string>("HGCEErechitCollection") ),
  hefRechitCollection_( ps.getParameter<std::string>("HGCHEFrechitCollection") ),
  hebRechitCollection_( ps.getParameter<std::string>("HGCHEBrechitCollection") ) {  
  produces< HGCeeRecHitCollection >(eeRechitCollection_);
  produces< HGChefRecHitCollection >(hefRechitCollection_);
  produces< HGChebRecHitCollection >(hebRechitCollection_);
  
  const std::string& componentType = ps.getParameter<std::string>("algo");
  worker_.reset( HGCalRecHitWorkerFactory::get()->create(componentType, ps) );
}

HGCalRecHitProducer::~HGCalRecHitProducer() {
}

void
HGCalRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;

  Handle< HGCeeUncalibratedRecHitCollection > pHGCeeUncalibRecHits;
  Handle< HGChefUncalibratedRecHitCollection > pHGChefUncalibRecHits;
  Handle< HGChebUncalibratedRecHitCollection > pHGChebUncalibRecHits;
  
  const HGCeeUncalibratedRecHitCollection*  eeUncalibRecHits = nullptr;
  const HGChefUncalibratedRecHitCollection*  hefUncalibRecHits = nullptr; 
  const HGChebUncalibratedRecHitCollection*  hebUncalibRecHits = nullptr; 

  // get the HGC uncalib rechit collection
  evt.getByToken( eeUncalibRecHitCollection_, pHGCeeUncalibRecHits);
  eeUncalibRecHits = pHGCeeUncalibRecHits.product();
    
  evt.getByToken( hefUncalibRecHitCollection_, pHGChefUncalibRecHits);
  hefUncalibRecHits = pHGChefUncalibRecHits.product();
  
  evt.getByToken( hebUncalibRecHitCollection_, pHGChebUncalibRecHits);
  hebUncalibRecHits = pHGChebUncalibRecHits.product();
    
  // collection of rechits to put in the event
  auto eeRecHits = std::make_unique<HGCeeRecHitCollection>();
  auto hefRecHits = std::make_unique<HGChefRecHitCollection>();
  auto hebRecHits = std::make_unique<HGChebRecHitCollection>();
    
  worker_->set(es);
  
  // loop over uncalibrated rechits to make calibrated ones
  for(auto it  = eeUncalibRecHits->begin(); it != eeUncalibRecHits->end(); ++it) {
    worker_->run(evt, *it, *eeRecHits);
  }
  
  // loop over uncalibrated rechits to make calibrated ones
  for(auto it  = hefUncalibRecHits->begin(); it != hefUncalibRecHits->end(); ++it) {
    worker_->run(evt, *it, *hefRecHits);
  }
  
  // loop over uncalibrated rechits to make calibrated ones
  for(auto it  = hebUncalibRecHits->begin(); it != hebUncalibRecHits->end(); ++it) {
    worker_->run(evt, *it, *hebRecHits);
  }
    
  // sort collections before attempting recovery, to avoid insertion of double recHits
  eeRecHits->sort();
  hefRecHits->sort();
  hebRecHits->sort();
    
  // put the collection of recunstructed hits in the event   
  LogInfo("HGCalRecHitInfo") << "total # HGCee calibrated rechits: " << eeRecHits->size();
  LogInfo("HGCalRecHitInfo") << "total # HGChef calibrated rechits: " << hefRecHits->size();
  LogInfo("HGCalRecHitInfo") << "total # HGCheb calibrated rechits: " << hebRecHits->size();
  
  evt.put(std::move(eeRecHits), eeRechitCollection_);
  evt.put(std::move(hefRecHits), hefRechitCollection_);
  evt.put(std::move(hebRecHits), hebRechitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( HGCalRecHitProducer );
