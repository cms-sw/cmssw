//
// Package:    LogErrorHarvester
// Class:      LogErrorHarvester

/**\class LogErrorHarvester LogErrorHarvester.cc FWCore/Modules/src/LogErrorHarvester.cc

 Description: Harvestes LogError messages and puts them into the Event

 Implementation:
     This simple implementation writes the std::vector<ErrorSummaryEntry> in the event,
     without any fancy attempt of encoding the strings or mapping them to ints
*/
//
// Original Author:  Giovanni Petrucciani
//         Created:  Thu Dec  4 16:22:40 CET 2008
//

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// system include files
#include <memory>

//
// class decleration
//

namespace edm {
  class LogErrorHarvester : public EDProducer {
    public:
      explicit LogErrorHarvester(ParameterSet const&);
      ~LogErrorHarvester();
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      virtual void beginJob();
      virtual void produce(Event&, EventSetup const&);
      virtual void endJob() ;
  };

  LogErrorHarvester::LogErrorHarvester(ParameterSet const&) {
     produces<std::vector<ErrorSummaryEntry> >();
  }

  LogErrorHarvester::~LogErrorHarvester() {
  }

  void
  LogErrorHarvester::produce(Event& iEvent, EventSetup const&) {
    if(!FreshErrorsExist()) {
      std::auto_ptr<std::vector<ErrorSummaryEntry> > errors(new std::vector<ErrorSummaryEntry>());
      iEvent.put(errors);
    } else {
      std::auto_ptr<std::vector<ErrorSummaryEntry> > errors(new std::vector<ErrorSummaryEntry>(LoggedErrorsSummary()));
      iEvent.put(errors);
    }
  }

  // ------------ method called once each job just before starting event loop  ------------
  void
  LogErrorHarvester::beginJob() {
    EnableLoggedErrorsSummary();
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void
  LogErrorHarvester::endJob() {
    DisableLoggedErrorsSummary();
  }


  // ------------ method called once each job for validation  ------------
  void
  LogErrorHarvester::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    descriptions.add("logErrorHarvester", desc);
  }
}

//define this as a plug-in
using edm::LogErrorHarvester;
DEFINE_FWK_MODULE(LogErrorHarvester);
