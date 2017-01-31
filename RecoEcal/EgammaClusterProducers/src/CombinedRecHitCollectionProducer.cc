#ifndef RECOECAL_EGAMMACLUSTERPRODUCERS_COMBINEDDRECHITCOLLECTIONPRODUCER_H
#define RECOECAL_EGAMMACLUSTERPRODUCERS_COMBINEDRECHITCOLLECTIONPRODUCER_H


//
//
// Class: CombinedRecHitCollectionProducer
// 
// 
// this class takes two rec-hit collections A and B and makes a new collection C
// collection C is identical to collection A except that if a rec-hit is also in B,
// the rec-hit from collection B is put into C instead of the A flavour
//
// takes inspiration from ReducedRecHitCollectionProducer


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include <iostream>

class CaloTopology;

class CombinedRecHitCollectionProducer : public edm::stream::EDProducer<> {
public:
  explicit CombinedRecHitCollectionProducer(const edm::ParameterSet&);
  ~CombinedRecHitCollectionProducer(){}
  virtual void produce(edm::Event &, const edm::EventSetup&);
  
private:
  
  const edm::EDGetTokenT<EcalRecHitCollection>  primaryRecHitsToken_; 
  const edm::EDGetTokenT<EcalRecHitCollection>  secondaryRecHitsToken_; 
  const std::vector<int> flagsToReplaceHit_;
  const std::string outputCollectionName_;
  const std::string outputReplacedHitsCollName_;
  const std::string outputReplacingHitsCollName_;
  const std::string outputDetIdCollName_;
};

CombinedRecHitCollectionProducer::
CombinedRecHitCollectionProducer(const edm::ParameterSet& iConfig):
  
  primaryRecHitsToken_(consumes<EcalRecHitCollection>(iConfig.getParameter< edm::InputTag > ("primaryRecHits"))),
  secondaryRecHitsToken_(consumes<EcalRecHitCollection>(iConfig.getParameter< edm::InputTag > ("secondaryRecHits"))),
  flagsToReplaceHit_(StringToEnumValue<EcalRecHit::Flags>(iConfig.getParameter<std::vector<std::string> >("flagsToReplaceHit"))),
  outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
  outputReplacedHitsCollName_(iConfig.getParameter<std::string>("outputReplacedHitsCollName")),
  outputReplacingHitsCollName_(iConfig.getParameter<std::string>("outputReplacingHitsCollName")),
  outputDetIdCollName_("hitsNotReplaced")
{

   //register your products
  produces< EcalRecHitCollection > (outputCollectionName_);
  produces< EcalRecHitCollection > (outputReplacedHitsCollName_);
  produces< EcalRecHitCollection > (outputReplacingHitsCollName_);
  produces< DetIdCollection> (outputDetIdCollName_);
}


void CombinedRecHitCollectionProducer::
produce (edm::Event& iEvent, 
	 const edm::EventSetup& iSetup)
{
  edm::Handle<EcalRecHitCollection> primaryRecHits;
  edm::Handle<EcalRecHitCollection> secondaryRecHits;
  
  iEvent.getByToken(primaryRecHitsToken_,primaryRecHits);
  iEvent.getByToken(secondaryRecHitsToken_,secondaryRecHits);
  
  auto outColl = std::make_unique<EcalRecHitCollection>();
  auto missingDetIds = std::make_unique<DetIdCollection>();
  auto outReplacedHits = std::make_unique<EcalRecHitCollection>(); //all the hits in primary coll replaced by secondary coll
  auto outReplacingHits = std::make_unique<EcalRecHitCollection>(); //all the hits in secondary coll which replaced a primary coll rec hit
  
  for(auto& hit : *primaryRecHits){
    
    //first we check if the hit as any flags set requiring it to be replaced
    
    if(hit.checkFlags(flagsToReplaceHit_)){

      auto secHit = secondaryRecHits->find(hit.detid());
      if(secHit!=secondaryRecHits->end()){ //found secondary hit
	outColl->push_back(*secHit);
	outReplacingHits->push_back(*secHit);
	outReplacedHits->push_back(hit);
      }else{
 	outColl->push_back(hit);
	missingDetIds->push_back(hit.detid());
      }
    }else{
      outColl->push_back(hit);
    }
  }
  iEvent.put(std::move(outColl),outputCollectionName_);
  iEvent.put(std::move(outReplacingHits),outputReplacingHitsCollName_);
  iEvent.put(std::move(outReplacedHits),outputReplacedHitsCollName_);
  iEvent.put(std::move(missingDetIds),outputDetIdCollName_);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CombinedRecHitCollectionProducer);

#endif
