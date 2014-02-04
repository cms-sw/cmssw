#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FastSimulation/CaloRecHitsProducer/interface/CaloRecHitsProducer.h"
#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "FastSimulation/CaloRecHitsProducer/interface/EcalBarrelRecHitsMaker.h"
#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"
#include "FastSimulation/CaloRecHitsProducer/interface/EcalPreshowerRecHitsMaker.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"

// Random engine
#include "FastSimulation/Utilities/interface/RandomEngine.h"

//#include <iostream>

CaloRecHitsProducer::CaloRecHitsProducer(edm::ParameterSet const & p)
  : EcalPreshowerRecHitsMaker_(NULL),EcalBarrelRecHitsMaker_(NULL),  EcalEndcapRecHitsMaker_(NULL), HcalRecHitsMaker_(NULL)
{    

  // Initialize the random number generator service
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration")
      << "CaloRecHitsProducer requires the RandomGeneratorService\n"
         "which is not present in the configuration file.\n"
         "You must add the service in the configuration file\n"
         "or remove the module that requires it";
  }
  random = new RandomEngine(&(*rng));
  theInputRecHitCollectionTypes = p.getParameter<std::vector<unsigned> >("InputRecHitCollectionTypes");
  theOutputRecHitCollections = p.getParameter<std::vector<std::string> >("OutputRecHitCollections");
  doDigis_ = p.getParameter<bool>("doDigis");
  doMiscalib_ = p.getParameter<bool>("doMiscalib");  
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("RecHitsFactory");
    
  for ( unsigned input=0; input<theInputRecHitCollectionTypes.size(); ++input ) { 

    switch ( theInputRecHitCollectionTypes[input] ) { 
      
    case 1: 
      {
	//Preshower
	if (theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size()) 
	  produces<ESRecHitCollection>(theOutputRecHitCollections[input]);
	else
	  produces<ESRecHitCollection>();

	if (doDigis_) 
	  std::cout << " The digitization of the preshower is not implemented " << std::endl;
	
	EcalPreshowerRecHitsMaker_ =  new EcalPreshowerRecHitsMaker(RecHitsParameters,random);  
      }
      break;
      
    case 2:
      { 
	//Ecal Barrel 
	if (theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size()) 
	  produces<EBRecHitCollection>(theOutputRecHitCollections[input]);
	else
	  produces<EBRecHitCollection>();
	
	if (doDigis_)  produces<EBDigiCollection>();
	EcalBarrelRecHitsMaker_ =  new EcalBarrelRecHitsMaker(RecHitsParameters,random);
      }
      break;
      
    case 3:
      { 
	//EcalEndcap
	if (theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size()) 
	  produces<EERecHitCollection>(theOutputRecHitCollections[input]);
	else
	  produces<EERecHitCollection>();
	if (doDigis_) produces<EEDigiCollection>();
	EcalEndcapRecHitsMaker_ =  new EcalEndcapRecHitsMaker(RecHitsParameters,random);
      }
      break;
      
    case 4:
      { 
	//HBHE
	if (theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size()) 
	    produces<HBHERecHitCollection>(theOutputRecHitCollections[input]);
	else
	    produces<HBHERecHitCollection>();
	
	if (doDigis_) produces<HBHEDigiCollection>();
	HcalRecHitsMaker_ =  new HcalRecHitsMaker(RecHitsParameters,4,random);
      }
      break;
      
    case 5:
      { 
	//HO
	if (theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size()) 
	  produces<HORecHitCollection>(theOutputRecHitCollections[input]);
	else
	  produces<HORecHitCollection>();

	if (doDigis_)  produces<HODigiCollection>();

	HcalRecHitsMaker_ =  new HcalRecHitsMaker(RecHitsParameters,5,random);
      }
      break;
      
    case 6:
      { 
	//HF
	if (theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size()) 
	  produces<HFRecHitCollection>(theOutputRecHitCollections[input]);
	else
	  produces<HFRecHitCollection>();	
	if(doDigis_)   produces<HFDigiCollection>();

	HcalRecHitsMaker_ =  new HcalRecHitsMaker(RecHitsParameters,6,random);
      }
      break;
      
    default:
      // Should not happen
      break;
      
    }
    
  }
 
}

CaloRecHitsProducer::~CaloRecHitsProducer() 
{ 
  if ( random ) { 
    delete random;
  }
}

void 
CaloRecHitsProducer::beginRun(const edm::Run & run, const edm::EventSetup & es) {

  for ( unsigned input=0; input<theInputRecHitCollectionTypes.size(); ++input ) { 
    switch ( theInputRecHitCollectionTypes[input] ) {       
    case 1: 
      {
	// preshower
	EcalPreshowerRecHitsMaker_->init(es); 	
      }
      break;
    case 2:
      {
	// ecal barrel
	EcalBarrelRecHitsMaker_->init(es,doDigis_,doMiscalib_);
      }
      break;
    case 3:
      {
	// ecal endcap
	EcalEndcapRecHitsMaker_->init(es,doDigis_,doMiscalib_);
      }
      break;
    case 4:
    case 5:
    case 6:
      {
	HcalRecHitsMaker_->init(es,doDigis_,doMiscalib_);
      }
      break;
    default:
      break;
    }
  }
}

void CaloRecHitsProducer::endJob()
{ 
  //std::cout << " (Fast)RecHitsProducer terminating " << std::endl; 
  if (EcalBarrelRecHitsMaker_) delete EcalBarrelRecHitsMaker_;
  if (EcalEndcapRecHitsMaker_) delete EcalEndcapRecHitsMaker_;
  if (EcalPreshowerRecHitsMaker_) delete EcalPreshowerRecHitsMaker_;
  if (HcalRecHitsMaker_) delete HcalRecHitsMaker_; 
}

void CaloRecHitsProducer::produce(edm::Event & iEvent, const edm::EventSetup & es)
{
   edm::ESHandle<HcalTopology> topo;
   es.get<HcalRecNumberingRecord>().get( topo );


  // create empty outputs for HCAL 
  // see RecoLocalCalo/HcalRecProducers/src/HcalSimpleReconstructor.cc
  for ( unsigned input=0; input<theInputRecHitCollectionTypes.size(); ++input ) { 
    switch ( theInputRecHitCollectionTypes[input] ) {       
    case 1: 
      {
	// preshower
	std::auto_ptr<ESRecHitCollection> reces(new ESRecHitCollection);  // ECAL pre-shower
	EcalPreshowerRecHitsMaker_->loadEcalPreshowerRecHits(iEvent,*reces);
	if ( theOutputRecHitCollections.size()&& theOutputRecHitCollections[input].size()) 
	  iEvent.put(reces,theOutputRecHitCollections[input]);
	else
	  iEvent.put(reces);
	break;
      }

    case 2:
      {
	// ecal barrel
	std::auto_ptr<EBRecHitCollection> receb(new EBRecHitCollection);  // ECAL Barrel
	std::auto_ptr<EBDigiCollection> digieb(new EBDigiCollection(1));
	EcalBarrelRecHitsMaker_->loadEcalBarrelRecHits(iEvent,*receb,*digieb);
	//	std::cout << " ECALBarrel " << receb->size() << std::endl;
	if ( theOutputRecHitCollections.size()&&theOutputRecHitCollections[input].size())
	  iEvent.put(receb,theOutputRecHitCollections[input]);
	else
	  iEvent.put(receb);

	if(doDigis_)
	  iEvent.put(digieb);	  
      }
      break;
    case 3:
      {
	// ecal endcap
	std::auto_ptr<EERecHitCollection> recee(new EERecHitCollection);  // ECAL Endcap
	std::auto_ptr<EEDigiCollection> digiee(new EEDigiCollection(1));
	EcalEndcapRecHitsMaker_->loadEcalEndcapRecHits(iEvent,*recee,*digiee);
	//	std::cout << " ECALEndcap " << recee->size() << std::endl;
	if ( theOutputRecHitCollections.size()&& theOutputRecHitCollections[input].size())
	  iEvent.put(recee,theOutputRecHitCollections[input]);
	else
	  iEvent.put(recee);

	if(doDigis_)
	  iEvent.put(digiee);
      }
      break;
    case 4:
      {
	// hbhe
	std::auto_ptr<HBHERecHitCollection> rec1(new HBHERecHitCollection); // Barrel+Endcap
	std::auto_ptr<HBHEDigiCollection> digihbhe(new HBHEDigiCollection);
	HcalRecHitsMaker_->loadHcalRecHits(iEvent,(*topo),*rec1,*digihbhe);
	if ( theOutputRecHitCollections.size()&& theOutputRecHitCollections[input].size())		    
	  iEvent.put(rec1,theOutputRecHitCollections[input]);
	else
	  iEvent.put(rec1);

	if(doDigis_)
	  iEvent.put(digihbhe);
      }
      break;
    case 5:
      {
	//ho
	std::auto_ptr<HORecHitCollection> rec2(new HORecHitCollection);     // Outer
	std::auto_ptr<HODigiCollection> digiho(new HODigiCollection);

	HcalRecHitsMaker_->loadHcalRecHits(iEvent,(*topo),*rec2,*digiho);
	if(theOutputRecHitCollections.size()&& theOutputRecHitCollections[input].size())	  
	  iEvent.put(rec2,theOutputRecHitCollections[input]);
	else
	  iEvent.put(rec2);
	if(doDigis_)
	  iEvent.put(digiho);
      }
      break;
    case 6:
      {
	//hf 
	std::auto_ptr<HFRecHitCollection> rec3(new HFRecHitCollection);     // Forward
	std::auto_ptr<HFDigiCollection> digihf(new HFDigiCollection);
	HcalRecHitsMaker_->loadHcalRecHits(iEvent,(*topo),*rec3,*digihf);
	if(theOutputRecHitCollections.size()&& theOutputRecHitCollections[input].size())
	  iEvent.put(rec3,theOutputRecHitCollections[input]);
	else
	  iEvent.put(rec3);
	if(doDigis_)
	  iEvent.put(digihf);	  
      }
      break;
    default:
      break;
    }
  }
}

DEFINE_FWK_MODULE(CaloRecHitsProducer);
