// -*- C++ -*-
//
// Package:    EcalBasicUncalibRecHitFilter
// Class:      EcalBasicUncalibRecHitFilter
// 
/**\class EcalBasicUncalibRecHitFilter EcalBasicUncalibRecHitFilter.cc Work/EcalBasicUncalibRecHitFilter/src/EcalBasicUncalibRecHitFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Wed Sep 19 16:21:29 CEST 2007
// $Id: EcalBasicUncalibRecHitFilter.cc,v 1.1 2007/11/27 15:28:00 meridian Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//
// class declaration
//

class EcalBasicUncalibRecHitFilter : public edm::EDFilter {
   public:
      explicit EcalBasicUncalibRecHitFilter(const edm::ParameterSet&);
      ~EcalBasicUncalibRecHitFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag EcalUncalibRecHitCollection_;
  double minAdc_;
  std::vector<int> maskedList_;

};

//
// constructors and destructor
//
EcalBasicUncalibRecHitFilter::EcalBasicUncalibRecHitFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  minAdc_     = iConfig.getUntrackedParameter<double>("adcCut", 12);
  maskedList_ = iConfig.getUntrackedParameter<std::vector<int> >("maskedChannels", maskedList_); //this is using the ashed index
  EcalUncalibRecHitCollection_ = iConfig.getParameter<edm::InputTag>("EcalUncalibRecHitCollection");
}


EcalBasicUncalibRecHitFilter::~EcalBasicUncalibRecHitFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
EcalBasicUncalibRecHitFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // getting very basic uncalRH
   Handle<EcalUncalibratedRecHitCollection> crudeHits;
   try {
     iEvent.getByLabel(EcalUncalibRecHitCollection_, crudeHits);
   } catch ( std::exception& ex) {
     LogWarning("EcalBasicUncalibRecHitFilter") << EcalUncalibRecHitCollection_ << " not available";
   }

   
   bool thereIsSignal = false;  
   // loop on crude rechits
   for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = crudeHits->begin(); hitItr != crudeHits->end(); ++hitItr ) {
     
     EcalUncalibratedRecHit hit = (*hitItr);
     
     // masking noisy channels
     std::vector<int>::iterator result;
     result = find( maskedList_.begin(), maskedList_.end(), EBDetId(hit.id()).hashedIndex() );    
     if  (result != maskedList_.end()) 
       // LogWarning("EcalFilter") << "skipping uncalRecHit for channel: " << ic << " with amplitude " << ampli_ ;
       continue; 
     
     float ampli_ = hit.amplitude();
     
     // seeking channels with signal and displaced jitter
     if (ampli_ >= minAdc_  ) 
       {
	 thereIsSignal = true;
	 // LogWarning("EcalFilter")  << "at evet: " << iEvent.id().event() 
	 // 				       << " and run: " << iEvent.id().run() 
	 // 				       << " there is OUT OF TIME signal at chanel: " << ic 
	 // 				       << " with amplitude " << ampli_  << " and max at: " << jitter_;
	 break;
       }
     
   }
   
   return thereIsSignal;
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalBasicUncalibRecHitFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalBasicUncalibRecHitFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalBasicUncalibRecHitFilter);
