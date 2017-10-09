// -*- C++ -*-
//
// Package:    Modules
// Class:      AbortOnEventIDAnalyzer
//
/**\class AbortOnEventIDAnalyzer AbortOnEventIDAnalyzer.cc FWCore/Modules/src/AbortOnEventIDAnalyzer.cc

 Description: Does a system abort when it seens the specified EventID. Useful for testing error handling.

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
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// system include files
#include <algorithm>
#include <memory>
#include <vector>
#include <stdlib.h>

//
// class decleration
//

class AbortOnEventIDAnalyzer : public edm::EDAnalyzer {
public:
   explicit AbortOnEventIDAnalyzer(edm::ParameterSet const&);
   ~AbortOnEventIDAnalyzer();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
   virtual void beginJob() override;
   virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
   virtual void endJob() override;

   // ----------member data ---------------------------
   std::vector<edm::EventID> ids_;
   bool throwException_;
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
AbortOnEventIDAnalyzer::AbortOnEventIDAnalyzer(edm::ParameterSet const& iConfig) :
  ids_(iConfig.getUntrackedParameter<std::vector<edm::EventID> >("eventsToAbort")),
  throwException_(iConfig.getUntrackedParameter<bool>("throwExceptionInsteadOfAbort"))
{
   //now do what ever initialization is needed

}


AbortOnEventIDAnalyzer::~AbortOnEventIDAnalyzer() {

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

namespace {
   struct CompareWithoutLumi {
      CompareWithoutLumi(edm::EventID const& iThis) : m_this(iThis) {
      }
      bool operator()(edm::EventID const& iOther) {
         return m_this.run() == iOther.run() && m_this.event() == iOther.event();
      }
      edm::EventID m_this;
   };
}

// ------------ method called to for each event  ------------
void
AbortOnEventIDAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
  std::vector<edm::EventID>::iterator itFind= std::find_if(ids_.begin(), ids_.end(), CompareWithoutLumi(iEvent.id()));
  if(itFind != ids_.end()) {
    if (throwException_) {
      throw cms::Exception("AbortEvent") << "Found event " << iEvent.id() << "\n";
    } else {
      abort();
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void
AbortOnEventIDAnalyzer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void
AbortOnEventIDAnalyzer::endJob() {
}

// ------------ method called once each job for validation
void
AbortOnEventIDAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::EventID> >("eventsToAbort");
  desc.addUntracked<bool>("throwExceptionInsteadOfAbort", false);
  descriptions.add("abortOnEventID", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(AbortOnEventIDAnalyzer);
