
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FastSimulation/CaloRecHitsProducer/interface/CaloRecHitCopy.h"

#include <iostream>

CaloRecHitCopy::CaloRecHitCopy(edm::ParameterSet const & p)

{    

  theInputRecHitCollectionTypes = p.getParameter<std::vector<unsigned> >("InputRecHitCollectionTypes");
  theInputRecHitCollections = p.getParameter<std::vector<edm::InputTag> >("InputRecHitCollections");
  theOutputRecHitCollections = p.getParameter<std::vector<std::string> >("OutputRecHitCollections");

  theOutputRecHitInstances.resize(theInputRecHitCollectionTypes.size());
  
  for ( unsigned input=0; input<theInputRecHitCollectionTypes.size(); ++input ) { 

    theOutputRecHitInstances[input] = 
      theOutputRecHitCollections[input] == "none" ?
      false : true;

    switch ( theInputRecHitCollectionTypes[input] ) { 
      
    case 1: 
      {
	//Preshower
	if ( !theOutputRecHitInstances[input] ) 
	  produces<ESRecHitCollection>();
	else
	  produces<ESRecHitCollection>(theOutputRecHitCollections[input]);
      }
      break;
      
    case 2:
      { 
	//Ecal Barrel 
	if ( !theOutputRecHitInstances[input] ) 
	  produces<EBRecHitCollection>();
	else
	  produces<EBRecHitCollection>(theOutputRecHitCollections[input]);
      }
      break;
      
    case 3:
      { 
	//EcalEndcap
	if ( !theOutputRecHitInstances[input] ) 
	  produces<EERecHitCollection>();
	else
	  produces<EERecHitCollection>(theOutputRecHitCollections[input]);
      }
      break;
      
    case 4:
      { 
	//HCAL
	if ( !theOutputRecHitInstances[input] ) 
	  produces<HBHERecHitCollection>();
	else
	  produces<HBHERecHitCollection>(theOutputRecHitCollections[input]);
      }
      break;
      
    case 5:
      { 
	//HO
	if ( !theOutputRecHitInstances[input] ) 
	  produces<HORecHitCollection>();
	else
	  produces<HORecHitCollection>(theOutputRecHitCollections[input]);
      }
      break;
      
    case 6:
      { 
	//HF
	if ( !theOutputRecHitInstances[input] ) 
	  produces<HFRecHitCollection>();
	else
	  produces<HFRecHitCollection>(theOutputRecHitCollections[input]);
      }
      break;
      
    default:
      // Should not happen
      break;
      
    }
    
  }
  

}

CaloRecHitCopy::~CaloRecHitCopy() { }

void 
CaloRecHitCopy::produce(edm::Event & iEvent, const edm::EventSetup & es)
{


  for ( unsigned input=0; input<theInputRecHitCollectionTypes.size(); ++input ) { 

    switch ( theInputRecHitCollectionTypes[input] ) { 
      
    case 1: 
      {
	//Preshower
	std::auto_ptr< ESRecHitCollection > copiedESRecHitCollection( new ESRecHitCollection );
	edm::Handle<ESRecHitCollection> ESRecHits;
	iEvent.getByLabel(theInputRecHitCollections[input],ESRecHits);
	ESRecHitCollection::const_iterator itES = ESRecHits->begin();
	ESRecHitCollection::const_iterator lastES = ESRecHits->end();
	// saves a bit of CPU
	copiedESRecHitCollection->reserve(ESRecHits->size());
	for ( ; itES!=lastES; ++itES++ ) {
	  EcalRecHit aHit(*itES);
	  copiedESRecHitCollection->push_back(aHit);
	}
	if ( !theOutputRecHitInstances[input] ) 
	  iEvent.put(copiedESRecHitCollection);
	else
	  iEvent.put(copiedESRecHitCollection,theOutputRecHitCollections[input]);
      }
      break;
      
    case 2: 
      {
	//Ecal Barrel 
	std::auto_ptr< EBRecHitCollection > copiedEBRecHitCollection( new EBRecHitCollection );
	edm::Handle<EBRecHitCollection> EBRecHits;
	iEvent.getByLabel(theInputRecHitCollections[input],EBRecHits);
	EBRecHitCollection::const_iterator itEB = EBRecHits->begin();
	EBRecHitCollection::const_iterator lastEB = EBRecHits->end();
	//saves a bit of CPU
	copiedEBRecHitCollection->reserve(EBRecHits->size());

	for ( ; itEB!=lastEB; ++itEB++ ) {
	  EcalRecHit aHit(*itEB);
	  copiedEBRecHitCollection->push_back(aHit);
	}
	if ( !theOutputRecHitInstances[input] ) 
	  iEvent.put(copiedEBRecHitCollection);
	else
	  iEvent.put(copiedEBRecHitCollection,theOutputRecHitCollections[input]);
      }
      break;
      
    case 3:
      {
	//EcalEndcap
	std::auto_ptr< EERecHitCollection > copiedEERecHitCollection( new EERecHitCollection );
	edm::Handle<EERecHitCollection> EERecHits;
	iEvent.getByLabel(theInputRecHitCollections[input],EERecHits);
	EERecHitCollection::const_iterator itEE = EERecHits->begin();
	EERecHitCollection::const_iterator lastEE = EERecHits->end();
	//saves a bit of CPU
	copiedEERecHitCollection->reserve(EERecHits->size());

	for ( ; itEE!=lastEE; ++itEE++ ) {
	  EcalRecHit aHit(*itEE);
	  copiedEERecHitCollection->push_back(aHit);
	}
	if ( !theOutputRecHitInstances[input] ) 
	  iEvent.put(copiedEERecHitCollection);
	else
	  iEvent.put(copiedEERecHitCollection,theOutputRecHitCollections[input]);
      }
      break;
      
    case 4:
      {
	//HCAL
	std::auto_ptr< HBHERecHitCollection > copiedHBHERecHitCollection( new HBHERecHitCollection );
	edm::Handle<HBHERecHitCollection> HBHERecHits;
	iEvent.getByLabel(theInputRecHitCollections[input],HBHERecHits);
	HBHERecHitCollection::const_iterator itHBHE = HBHERecHits->begin();
	HBHERecHitCollection::const_iterator lastHBHE = HBHERecHits->end();
	//saves a bit of CPU
	copiedHBHERecHitCollection->reserve(HBHERecHits->size());

	for ( ; itHBHE!=lastHBHE; ++itHBHE++ ) {
	  HBHERecHit aHit(*itHBHE);
	  copiedHBHERecHitCollection->push_back(aHit);
	}
	if ( !theOutputRecHitInstances[input] ) 
	  iEvent.put(copiedHBHERecHitCollection);
	else
	  iEvent.put(copiedHBHERecHitCollection,theOutputRecHitCollections[input]);
      }
      break;
      
    case 5:
      {
	//HO
	std::auto_ptr< HORecHitCollection > copiedHORecHitCollection( new HORecHitCollection );
	edm::Handle<HORecHitCollection> HORecHits;
	iEvent.getByLabel(theInputRecHitCollections[input],HORecHits);
	HORecHitCollection::const_iterator itHO = HORecHits->begin();
	HORecHitCollection::const_iterator lastHO = HORecHits->end();
	//saves a bit of CPU
	copiedHORecHitCollection->reserve(HORecHits->size());

	for ( ; itHO!=lastHO; ++itHO++ ) {
	  HORecHit aHit(*itHO);
	  copiedHORecHitCollection->push_back(aHit);
	}
	if ( !theOutputRecHitInstances[input] ) 
	  iEvent.put(copiedHORecHitCollection);
	else
	  iEvent.put(copiedHORecHitCollection,theOutputRecHitCollections[input]);
      }
      break;
      
    case 6:
      {
	//HF
	std::auto_ptr< HFRecHitCollection > copiedHFRecHitCollection( new HFRecHitCollection );
	edm::Handle<HFRecHitCollection> HFRecHits;
	iEvent.getByLabel(theInputRecHitCollections[input],HFRecHits);
	HFRecHitCollection::const_iterator itHF = HFRecHits->begin();
	HFRecHitCollection::const_iterator lastHF = HFRecHits->end();
	//saves a bit of CPU
	copiedHFRecHitCollection->reserve(HFRecHits->size());
	
	for ( ; itHF!=lastHF; ++itHF++ ) {
	  HFRecHit aHit(*itHF);
	  copiedHFRecHitCollection->push_back(aHit);
	}
	if ( !theOutputRecHitInstances[input] ) 
	  iEvent.put(copiedHFRecHitCollection);
	else
	  iEvent.put(copiedHFRecHitCollection,theOutputRecHitCollections[input]);
      }
      break;
      
    default:
      // Should not happen
      break;
      
    }
    
  }

}

DEFINE_FWK_MODULE(CaloRecHitCopy);
