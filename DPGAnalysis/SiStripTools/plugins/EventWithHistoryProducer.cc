// -*- C++ -*-
//
// Package:    EventWithHistoryProducer
// Class:      EventWithHistoryProducer
// 
/**\class EventWithHistoryProducer EventWithHistoryProducer.cc DPGAnalysis/SiStripTools/plugins/EventWithHistoryProducer.cc

 Description: EDProducer of EventWithHistory which rely on the presence of the previous event in the analyzed dataset

 Implementation:
     
*/
//
// Original Author:  Andrea Venturi
//         Created:  Sun Nov 30 19:05:41 CET 2008
// $Id: EventWithHistoryProducer.cc,v 1.3 2013/02/27 19:49:46 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "DPGAnalysis/SiStripTools/interface/TinyEvent.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"

//
// class decleration
//

class EventWithHistoryProducer : public edm::EDProducer {
   public:
      explicit EventWithHistoryProducer(const edm::ParameterSet&);
      ~EventWithHistoryProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  const unsigned int _depth;
  EventWithHistory _prevHE;
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
EventWithHistoryProducer::EventWithHistoryProducer(const edm::ParameterSet& iConfig):
  _depth(iConfig.getUntrackedParameter<unsigned int>("historyDepth")),
  _prevHE() 
{

  produces<EventWithHistory>();

   //now do what ever other initialization is needed
  
}


EventWithHistoryProducer::~EventWithHistoryProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EventWithHistoryProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;

  std::auto_ptr<EventWithHistory> heOut(new EventWithHistory(iEvent.id().event(),iEvent.orbitNumber(),iEvent.bunchCrossing()));
  heOut->add(_prevHE,_depth);

  if(*heOut < _prevHE) edm::LogInfo("EventsNotInOrder") << " Events not in order " << _prevHE._event;

  _prevHE = *heOut;
  iEvent.put(heOut);
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
EventWithHistoryProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventWithHistoryProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventWithHistoryProducer);
