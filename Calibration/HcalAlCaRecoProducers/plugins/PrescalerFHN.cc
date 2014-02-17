// -*- C++ -*-
//
// Package:    HCALNoiseAlCaReco
// Class:      HCALNoiseAlCaReco
// 
/**\class HCALNoiseAlCaReco HCALNoiseAlCaReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kenneth Case Rossato
//         Created:  Wed Mar 25 13:05:10 CET 2008
// $Id: PrescalerFHN.cc,v 1.4 2010/02/16 22:24:16 wdd Exp $
//
//
// modified to PrecalerFHN by Grigory Safronov 27/03/09

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <string>

//
// class declaration
//

using namespace edm;

class PrescalerFHN : public edm::EDFilter {
   public:
      explicit PrescalerFHN(const edm::ParameterSet&);
      ~PrescalerFHN();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      // ----------member data ---------------------------

  void init(const edm::TriggerResults &,
            const edm::TriggerNames & triggerNames);

  edm::ParameterSetID triggerNamesID_;

  edm::InputTag triggerTag;

  std::map<std::string, unsigned int> prescales;
  std::map<std::string, unsigned int> prescale_counter;

  std::map<std::string, unsigned int> trigger_indices;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrescalerFHN::PrescalerFHN(const edm::ParameterSet& iConfig)
  : triggerTag(iConfig.getParameter<edm::InputTag>("TriggerResultsTag"))
{
   //now do what ever initialization is needed
  std::vector<edm::ParameterSet> prescales_in(iConfig.getParameter<std::vector<edm::ParameterSet> >("Prescales"));

  for (std::vector<edm::ParameterSet>::const_iterator cit = prescales_in.begin();
       cit != prescales_in.end(); cit++) {

    std::string name(cit->getParameter<std::string>("HLTName"));
    unsigned int factor(cit->getParameter<unsigned int>("PrescaleFactor"));

    // does some exception get thrown if parameters aren't available? should test...

    prescales[name] = factor;
    prescale_counter[name] = 0;
  }
  
}


PrescalerFHN::~PrescalerFHN()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void PrescalerFHN::init(const edm::TriggerResults &result,
                        const edm::TriggerNames & triggerNames)
{
  trigger_indices.clear();

  for (std::map<std::string, unsigned int>::const_iterator cit = prescales.begin();
       cit != prescales.end(); cit++) {

    trigger_indices[cit->first] = triggerNames.triggerIndex(cit->first);
    
    if (trigger_indices[cit->first] >= result.size()) {
      // trigger path not found
      LogDebug("") << "requested HLT path does not exist: " << cit->first;
      trigger_indices.erase(cit->first);
    }
    
  }
}

// ------------ method called on each new Event  ------------
bool
PrescalerFHN::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   /* Goal for this skim:
      Prescaling MET HLT paths
      Option to turn off HSCP filter
      - Doing that by treating it as an HLT with prescale 1
   */

   // Trying to mirror HLTrigger/HLTfilters/src/HLTHighLevel.cc where possible

   Handle<TriggerResults> trh;
   iEvent.getByLabel(triggerTag, trh);

   if (trh.isValid()) {
     LogDebug("") << "TriggerResults found, number of HLT paths: " << trh->size();
   } else {
     LogDebug("") << "TriggerResults product not found - returning result=false!";
     return false;
   }

   const edm::TriggerNames & triggerNames = iEvent.triggerNames(*trh);
   if (triggerNamesID_ != triggerNames.parameterSetID()) {
     triggerNamesID_ = triggerNames.parameterSetID();
     init(*trh, triggerNames);
   }

   // Trigger indices are ready at this point
   // - Begin checking for HLT bits

   bool accept_event = false;
   for (std::map<std::string, unsigned int>::const_iterator cit = trigger_indices.begin();
	cit != trigger_indices.end(); cit++) {
     if (trh->accept(cit->second)) {
       prescale_counter[cit->first]++;
       if (prescale_counter[cit->first] >= prescales[cit->first]) {
	 accept_event = true;
	 prescale_counter[cit->first] = 0;
       }
     }
   }

   return accept_event;
}

// ------------ method called once each job just before starting event loop  ------------
void 
PrescalerFHN::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PrescalerFHN::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrescalerFHN);
