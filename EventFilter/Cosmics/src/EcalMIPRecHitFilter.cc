// -*- C++ -*-
//
// Package:    EcalMIPRecHitFilter
// Class:      EcalMIPRecHitFilter
// 
/**\class EcalMIPRecHitFilter EcalMIPRecHitFilter.cc Work/EcalMIPRecHitFilter/src/EcalMIPRecHitFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Wed Sep 19 16:21:29 CEST 2007
// $Id: EcalMIPRecHitFilter.cc,v 1.1 2008/03/26 15:07:19 haupt Exp $
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

//
// class declaration
//

class EcalMIPRecHitFilter : public HLTFilter {
   public:
      explicit EcalMIPRecHitFilter(const edm::ParameterSet&);
      ~EcalMIPRecHitFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag EcalRecHitCollection_;
  double minAmp_;
  double minSingleAmp_;
  std::vector<int> maskedList_;
  int side_;

};

//
// constructors and destructor
//
EcalMIPRecHitFilter::EcalMIPRecHitFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  minSingleAmp_     = iConfig.getUntrackedParameter<double>("SingleAmpMin", 9);
  minAmp_     = iConfig.getUntrackedParameter<double>("AmpMin", 6.5);
  maskedList_ = iConfig.getUntrackedParameter<std::vector<int> >("maskedChannels", maskedList_); //this is using the ashed index
  EcalRecHitCollection_ = iConfig.getParameter<edm::InputTag>("EcalRecHitCollection");
  side_ = iConfig.getUntrackedParameter<int>("side", 3);  
}


EcalMIPRecHitFilter::~EcalMIPRecHitFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
EcalMIPRecHitFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // getting very basic uncalRH
   Handle<EcalRecHitCollection> recHits;
   //try {
   //  iEvent.getByLabel(EcalRecHitCollection_, recHits);
   //} catch ( std::exception& ex) {
   //  LogWarning("EcalMIPRecHitFilter") << EcalRecHitCollection_ << " not available";
   //}
   iEvent.getByLabel(EcalRecHitCollection_, recHits);
   if (!recHits.isValid()){
      LogWarning("EcalMIPRecHitFilter") << EcalRecHitCollection_ << " not available";
   }

   ESHandle<CaloTopology> caloTopo;
   iSetup.get<CaloTopologyRecord>().get(caloTopo);

   bool thereIsSignal = false;  
   // loop on  rechits
   for ( EcalRecHitCollection::const_iterator hitItr = recHits->begin(); hitItr != recHits->end(); ++hitItr ) {
     
     EcalRecHit hit = (*hitItr);
     
     // masking noisy channels //KEEP this for now, just in case a few show up
     std::vector<int>::iterator result;
     result = find( maskedList_.begin(), maskedList_.end(), EBDetId(hit.id()).hashedIndex() );    
     if  (result != maskedList_.end()) 
       // LogWarning("EcalFilter") << "skipping uncalRecHit for channel: " << ic << " with amplitude " << ampli_ ;
       continue; 
     
     float ampli_ = hit.energy();
     EBDetId ebDet = hit.id();
     
     // seeking channels with signal and displaced jitter
     if (ampli_ >= minSingleAmp_  ) 
       {
	 //std::cout << " THIS AMPLITUDE WORKS " << ampli_ << std::endl;
	 thereIsSignal = true;
	 // LogWarning("EcalFilter")  << "at evet: " << iEvent.id().event() 
	 // 				       << " and run: " << iEvent.id().run() 
	 // 				       << " there is OUT OF TIME signal at chanel: " << ic 
	 // 				       << " with amplitude " << ampli_  << " and max at: " << jitter_;
	 break;
       }

     //Check for more robust selection other than just single crystal cosmics
     if (ampli_ >= minAmp_)
       {
	 //std::cout << " THIS AMPLITUDE WORKS " << ampli_ << std::endl;
	  std::vector<DetId> neighbors = caloTopo->getWindow(ebDet,side_,side_);
          double secondMin = 0.;
	  double E9 = ampli_;
          for(std::vector<DetId>::const_iterator detitr = neighbors.begin(); detitr != neighbors.end(); ++detitr)
	    {
	      EcalRecHitCollection::const_iterator thishit = recHits->find((*detitr));
              if (thishit == recHits->end()) 
		{
		  LogWarning("EcalMIPRecHitFilter") << "No RecHit available, for "<< EBDetId(*detitr);
		  continue;
		}
	      double thisamp = (*thishit).energy();
              E9+=thisamp;
	      if (thisamp > secondMin) secondMin = thisamp;
	    }
          double E2 = ampli_ + secondMin;
          if (E2 > 2*minAmp_ ) 
	    {
	       thereIsSignal = true;
	       break;
	    }
	  if (E9 > 2*minAmp_ ) 
	    {
	       thereIsSignal = true;
	       break;
	    }

       }
     
   }
   //std::cout << " Ok is There one of THEM " << thereIsSignal << std::endl;
   return thereIsSignal;
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalMIPRecHitFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalMIPRecHitFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalMIPRecHitFilter);
