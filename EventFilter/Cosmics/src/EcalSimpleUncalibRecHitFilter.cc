// -*- C++ -*-
//
// Package:    EcalSimpleUncalibRecHitFilter
// Class:      EcalSimpleUncalibRecHitFilter
// 
/**\class EcalSimpleUncalibRecHitFilter EcalSimpleUncalibRecHitFilter.cc Work/EcalSimpleUncalibRecHitFilter/src/EcalSimpleUncalibRecHitFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Wed Sep 19 16:21:29 CEST 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//
// class declaration
//

class EcalSimpleUncalibRecHitFilter : public edm::EDFilter {
public:
  explicit EcalSimpleUncalibRecHitFilter(const edm::ParameterSet&);
  ~EcalSimpleUncalibRecHitFilter();

private:
  virtual bool filter(edm::Event &, edm::EventSetup const &) override;

  // ----------member data ---------------------------

  edm::InputTag     EcalUncalibRecHitCollection_;
  double            minAdc_;
  std::vector<int>  maskedList_;

};

//
// constructors and destructor
//
EcalSimpleUncalibRecHitFilter::EcalSimpleUncalibRecHitFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  minAdc_     = iConfig.getUntrackedParameter<double>("adcCut", 12);
  maskedList_ = iConfig.getUntrackedParameter<std::vector<int> >("maskedChannels", maskedList_); //this is using the ashed index
  EcalUncalibRecHitCollection_ = iConfig.getParameter<edm::InputTag>("EcalUncalibRecHitCollection");
}


EcalSimpleUncalibRecHitFilter::~EcalSimpleUncalibRecHitFilter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
EcalSimpleUncalibRecHitFilter::filter(edm::Event & iEvent, edm::EventSetup const & iSetup)
{
  using namespace edm;

  // getting very basic uncalRH
  Handle<EcalUncalibratedRecHitCollection> crudeHits;
  try {
    iEvent.getByLabel(EcalUncalibRecHitCollection_, crudeHits);
  } catch ( std::exception& ex) {
    LogWarning("EcalSimpleUncalibRecHitFilter") << EcalUncalibRecHitCollection_ << " not available";
  }


  bool thereIsSignal = false;
  // loop on crude rechits
  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = crudeHits->begin(); hitItr != crudeHits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);

    // masking noisy channels
    std::vector<int>::const_iterator result = std::find( maskedList_.begin(), maskedList_.end(), EBDetId(hit.id()).hashedIndex() );
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

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalSimpleUncalibRecHitFilter);
