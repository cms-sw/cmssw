// -*- C++ -*-
//
// Package:    Modules
// Class:      EventIDChecker
// 
/**\class EventIDChecker EventIDChecker.cc FWCore/Modules/src/EventIDChecker.cc

 Description: Checks that the events passed to it come in the order specified in its configuration

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 16 15:42:17 CDT 2009
// $Id$
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/EventID.h"

//
// class decleration
//

class EventIDChecker : public edm::EDAnalyzer {
public:
   explicit EventIDChecker(const edm::ParameterSet&);
   ~EventIDChecker();
   
   
private:
   virtual void beginJob() ;
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob() ;
   virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

   // ----------member data ---------------------------
   std::vector<edm::EventID> ids_;
   unsigned int index_;
   
   unsigned int multiProcessSequentialEvents_;
   unsigned int numberOfEventsToSkip_;
   unsigned int numberOfEventsLeftBeforeSkip_;
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
EventIDChecker::EventIDChecker(const edm::ParameterSet& iConfig):
ids_(iConfig.getUntrackedParameter<std::vector<edm::EventID> >("eventSequence")),
index_(0),
multiProcessSequentialEvents_(iConfig.getUntrackedParameter<unsigned int>("multiProcessSequentialEvents",0)),
numberOfEventsToSkip_(0),
numberOfEventsLeftBeforeSkip_(0)
{
   //now do what ever initialization is needed

}


EventIDChecker::~EventIDChecker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EventIDChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if(0!=numberOfEventsToSkip_) {
      ++numberOfEventsLeftBeforeSkip_;
      if(numberOfEventsLeftBeforeSkip_ > multiProcessSequentialEvents_) {
         numberOfEventsLeftBeforeSkip_=1;
         index_ += numberOfEventsToSkip_;
      }
   }
         
   if(index_ >= ids_.size()) {
      throw cms::Exception("TooManyEvents")<<"Was passes "<<ids_.size()<<" EventIDs but have processed more events than that\n";
   }
   if(iEvent.id() != ids_[index_]) {
      throw cms::Exception("WrongEvent")<<"Was expecting event "<<ids_[index_]<<" but was given "<<iEvent.id()<<"\n";
   }
   ++index_;
}


// ------------ method called once each job just before starting event loop  ------------
void 
EventIDChecker::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventIDChecker::endJob() {
}

void 
EventIDChecker::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
   numberOfEventsToSkip_ = (iNumberOfChildren-1)*multiProcessSequentialEvents_;
   index_ = iChildIndex*multiProcessSequentialEvents_;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventIDChecker);
