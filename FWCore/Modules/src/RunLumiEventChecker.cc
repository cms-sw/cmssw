// -*- C++ -*-
//
// Package:    Modules
// Class:      RunLumiEventChecker
//
/**\class RunLumiEventChecker RunLumiEventChecker.cc FWCore/Modules/src/RunLumiEventChecker.cc

 Description: Checks that the events passed to it come in the order specified in its configuration

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 16 15:42:17 CDT 2009
//
//

// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// system include files
#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <map>

//
// class decleration
//

class RunLumiEventChecker : public edm::EDAnalyzer {
public:
   explicit RunLumiEventChecker(edm::ParameterSet const&);
   ~RunLumiEventChecker() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
   void beginJob() override;
   void analyze(edm::Event const&, edm::EventSetup const&) override;
   void endJob() override;

   void beginRun(edm::Run const& run, edm::EventSetup const& es) override;
   void endRun(edm::Run const& run, edm::EventSetup const& es) override;
   
   void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;
   void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;

   void check(edm::EventID const& iID, bool isEvent);
   
   // ----------member data ---------------------------
   std::vector<edm::EventID> ids_;
   unsigned int index_;
   std::map<edm::EventID, unsigned int> seenIDs_;
   
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
RunLumiEventChecker::RunLumiEventChecker(edm::ParameterSet const& iConfig) :
  ids_(iConfig.getUntrackedParameter<std::vector<edm::EventID> >("eventSequence")),
  index_(0)
{
   //now do what ever initialization is needed
}


RunLumiEventChecker::~RunLumiEventChecker() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

void
RunLumiEventChecker::check(edm::EventID const& iEventID, bool iIsEvent) {
   if(index_ >= ids_.size()) {
      throw cms::Exception("TooManyEvents") << "Was passes " << ids_.size() << " EventIDs but have processed more events than that\n";
   }
   if(iEventID  != ids_[index_]) {
      throw cms::Exception("WrongEvent") << "Was expecting event " << ids_[index_] << " but was given " << iEventID << "\n";
   }
   ++index_;
}

// ------------ method called to for each event  ------------
void
RunLumiEventChecker::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
   check(iEvent.id(), true);
}

void 
RunLumiEventChecker::beginRun(edm::Run const& run, edm::EventSetup const&) {
   check(edm::EventID(run.id().run(), 0, 0), false);   
}
void 
RunLumiEventChecker::endRun(edm::Run const& run, edm::EventSetup const&) {
   check(edm::EventID(run.id().run(), 0, 0), false);   
}

void 
RunLumiEventChecker::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
   check(edm::EventID(lumi.id().run(), lumi.id().luminosityBlock(), 0), false);   
}

void 
RunLumiEventChecker::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
   check(edm::EventID(lumi.id().run(), lumi.id().luminosityBlock(), 0), false);   
}


// ------------ method called once each job just before starting event loop  ------------
void
RunLumiEventChecker::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void
RunLumiEventChecker::endJob() {
   if(index_ != ids_.size()) {
      throw cms::Exception("WrongNumberOfEvents")<<"Saw "<<index_<<" events but was supposed to see "<<ids_.size()<<"\n";
   }   
}

// ------------ method called once each job for validation
void
RunLumiEventChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::EventID> >("eventSequence");
  descriptions.add("runLumiEventIDChecker", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(RunLumiEventChecker);
