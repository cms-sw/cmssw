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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Provenance/interface/MinimalEventID.h"

//
// class decleration
//

class EventIDChecker : public edm::EDAnalyzer {
public:
   explicit EventIDChecker(edm::ParameterSet const&);
   ~EventIDChecker();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
   
   
private:
   virtual void beginJob() ;
   virtual void analyze(edm::Event const&, edm::EventSetup const&);
   virtual void endJob() ;
   virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

   // ----------member data ---------------------------
   std::vector<edm::MinimalEventID> ids_;
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
EventIDChecker::EventIDChecker(edm::ParameterSet const& iConfig) :
  ids_(iConfig.getUntrackedParameter<std::vector<edm::MinimalEventID> >("eventSequence")),
  index_(0),
  multiProcessSequentialEvents_(iConfig.getUntrackedParameter<unsigned int>("multiProcessSequentialEvents")),
  numberOfEventsToSkip_(0),
  numberOfEventsLeftBeforeSkip_(0) {
   //now do what ever initialization is needed

}


EventIDChecker::~EventIDChecker() {
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EventIDChecker::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
   if(0 != numberOfEventsToSkip_) {
      ++numberOfEventsLeftBeforeSkip_;
      if(numberOfEventsLeftBeforeSkip_ > multiProcessSequentialEvents_) {
         numberOfEventsLeftBeforeSkip_ = 1;
         index_ += numberOfEventsToSkip_;
      }
   }
         
   if(index_ >= ids_.size()) {
      throw cms::Exception("TooManyEvents")<<"Was passes "<<ids_.size()<<" EventIDs but have processed more events than that\n";
   }
   if(iEvent.id().run() != ids_[index_].run() || iEvent.id().event() != ids_[index_].event()) {
      throw cms::Exception("WrongEvent")<<"Was expecting event "<<ids_[index_]<<" but was given "<<iEvent.id()<<"\n";
   }
   ++index_;
}


// ------------ method called once each job just before starting event loop  ------------
void 
EventIDChecker::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventIDChecker::endJob() {
}

// ------------ method called once each job for validation
void
EventIDChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::MinimalEventID> >("eventSequence");
  desc.addUntracked<unsigned int>("multiProcessSequentialEvents", 0U);
  descriptions.add("PoolOutputModule", desc);
}

void 
EventIDChecker::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
   numberOfEventsToSkip_ = (iNumberOfChildren-1)*multiProcessSequentialEvents_;
   index_ = iChildIndex*multiProcessSequentialEvents_;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventIDChecker);
