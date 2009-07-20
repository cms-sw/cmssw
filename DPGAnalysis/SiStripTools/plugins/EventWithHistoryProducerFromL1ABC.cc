// -*- C++ -*-
//
// Package:    EventWithHistoryProducerFromL1ABC
// Class:      EventWithHistoryProducerFromL1ABC
// 
/**\class EventWithHistoryProducerFromL1ABC EventWithHistoryProducerFromL1ABC.cc DPGAnalysis/SiStripTools/src/EventWithHistoryProducerFromL1ABC.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Jun 30 15:26:20 CET 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
//
// class decleration
//

class EventWithHistoryProducerFromL1ABC : public edm::EDProducer {
   public:
      explicit EventWithHistoryProducerFromL1ABC(const edm::ParameterSet&);
      ~EventWithHistoryProducerFromL1ABC();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag _l1abccollection;

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
EventWithHistoryProducerFromL1ABC::EventWithHistoryProducerFromL1ABC(const edm::ParameterSet& iConfig):
  _l1abccollection(iConfig.getParameter<edm::InputTag>("l1ABCCollection"))
{
  produces<EventWithHistory>();
   
   //now do what ever other initialization is needed
  
}


EventWithHistoryProducerFromL1ABC::~EventWithHistoryProducerFromL1ABC()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EventWithHistoryProducerFromL1ABC::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<L1AcceptBunchCrossingCollection > pIn;
   iEvent.getByLabel(_l1abccollection,pIn);

   edm::LogInfo("L1ABCCollectionSize") << pIn->size() << " L1ABC found ";

   for(L1AcceptBunchCrossingCollection::const_iterator l1abc=pIn->begin();l1abc!=pIn->end();++l1abc) {
     edm::LogVerbatim("L1ABCDebug") << *l1abc;
   }


   std::auto_ptr<EventWithHistory> pOut(new EventWithHistory(iEvent,*pIn));
   iEvent.put(pOut);

 
}

// ------------ method called once each job just before starting event loop  ------------
void 
EventWithHistoryProducerFromL1ABC::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventWithHistoryProducerFromL1ABC::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventWithHistoryProducerFromL1ABC);
