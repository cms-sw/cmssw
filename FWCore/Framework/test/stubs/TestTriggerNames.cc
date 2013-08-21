
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace edm {
  class EventSetup;
}

using namespace edm;

namespace edmtest {

  class TestTriggerNames : public edm::EDAnalyzer {
  public:

    typedef std::vector<std::string> Strings;

    explicit TestTriggerNames(edm::ParameterSet const&);
    virtual ~TestTriggerNames();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
    void endJob();

  private:

    unsigned int iEvent_;
    Strings expected_trigger_paths_;
    Strings expected_trigger_previous_;
    Strings expected_end_paths_;
    bool streamerSource_;
    bool dumpPSetRegistry_;
    std::vector<unsigned int> expectedTriggerResultsHLT_;
    std::vector<unsigned int> expectedTriggerResultsPROD_;
  };

  // -----------------------------------------------------------------

  TestTriggerNames::TestTriggerNames(edm::ParameterSet const& ps) :
    iEvent_(0U),
    expected_trigger_paths_(ps.getUntrackedParameter<Strings>("trigPaths", Strings())),
    expected_trigger_previous_(ps.getUntrackedParameter<Strings>("trigPathsPrevious", Strings())),
    expected_end_paths_(ps.getUntrackedParameter<Strings>("endPaths", Strings())),
    streamerSource_(ps.getUntrackedParameter<bool>("streamerSource", false)),
    dumpPSetRegistry_(ps.getUntrackedParameter<bool>("dumpPSetRegistry", false)),
    expectedTriggerResultsHLT_(ps.getUntrackedParameter<std::vector<unsigned int> >("expectedTriggerResultsHLT", std::vector<unsigned int>())),
    expectedTriggerResultsPROD_(ps.getUntrackedParameter<std::vector<unsigned int> >("expectedTriggerResultsPROD", std::vector<unsigned int>())) {
      consumesMany<edm::TriggerResults>();
  }

  // -----------------------------------------------------------------

  TestTriggerNames::~TestTriggerNames() {
  }

  // -----------------------------------------------------------------

  void TestTriggerNames::analyze(edm::Event const& e, edm::EventSetup const&) {
    if(dumpPSetRegistry_) {
      pset::Registry* psetRegistry = pset::Registry::instance();
      psetRegistry->print(std::cout);
    }

    // Runs some tests on the TriggerNamesService
    if(expected_trigger_paths_.size() > 0) {

      Strings triggernames;
      edm::Service<edm::service::TriggerNamesService> tns;
      triggernames = tns->getTrigPaths();
      if(triggernames.size() != expected_trigger_paths_.size()) {
        std::cerr << "TestTriggerNames: "
             << "Expected and actual trigger path list not the same size" << std::endl;
        abort();
      }
      for(Strings::size_type i = 0; i < expected_trigger_paths_.size(); ++i) {
        if(triggernames[i] != expected_trigger_paths_[i]) {
          std::cerr << "TestTriggerNames: "
               << "Expected and actual trigger paths don't match" << std::endl;
          abort();
        }
      }
    }

    if(expected_end_paths_.size() > 0) {

      Strings endnames;
      edm::Service<edm::service::TriggerNamesService> tns;
      endnames = tns->getEndPaths();
      if(endnames.size() != expected_end_paths_.size()) {
        std::cerr << "TestTriggerNames: "
             << "Expected and actual end path list not the same size" << std::endl;
        abort();
      }
      for(Strings::size_type i = 0; i < expected_end_paths_.size(); ++i) {
        if(endnames[i] != expected_end_paths_[i]) {
          std::cerr << "TestTriggerNames: "
               << "Expected and actual end paths don't match" << std::endl;
          abort();
        }
      }
    }

    typedef std::vector<edm::Handle<edm::TriggerResults> > Trig;
    Trig prod;
    e.getManyByType(prod);

    if(expected_trigger_previous_.size() > 0) {

      if(prod.size() == 0) {
        std::cerr << "TestTriggerNames: "
             << "No TriggerResults object found, expected previous trigger results"
             << std::endl;
        abort();
      }

      // The TriggerResults object for this event has not been written into
      // the event yet. prod[0] is the TriggerResults for the most recent
      // previous process.  This assumes this is not run in an end path.

      Strings triggernames;
      edm::Service<edm::service::TriggerNamesService> tns;
      if(tns->getTrigPaths(*prod[0], triggernames)) {
        if(triggernames.size() != expected_trigger_previous_.size()) {
          std::cerr << "TestTriggerNames: "
               << "Expected and actual previous trigger path lists not the same size" << std::endl;
          abort();
        }
        for(Strings::size_type i = 0; i < expected_trigger_previous_.size(); ++i) {
          if(triggernames[i] != expected_trigger_previous_[i]) {
            std::cerr << "TestTriggerNames: "
                 << "Expected and actual previous trigger paths don't match" << std::endl;
            abort();
          }
        }
      }
      else {
        std::cerr << "TestTriggerNames: "
             << "Failed finding trigger names from a previous process" << std::endl;
        abort();
      }
      bool fromPSetRegistry;
      if(tns->getTrigPaths(*prod[0], triggernames, fromPSetRegistry)) {
        if(!fromPSetRegistry) {
          std::cerr << "TestTriggerNames: "
               << "fromPSetRegistry returned with incorrect value" << std::endl;
          abort();
        }
      }

      // The provenance of the TriggerResults object should also determine the
      // ID for the parameter set that lists the trigger paths.
      // Test this by getting this parameter set and verifying the trigger
      // paths are the correct size.
      if(!streamerSource_) {
        ParameterSet const& trigpset = parameterSet(*prod[0].provenance());
        Strings trigpaths = trigpset.getParameter<Strings>("@trigger_paths");
        if(trigpaths.size() != expected_trigger_previous_.size()) {
          std::cerr << "TestTriggerNames: Using provenance\n"
                    << "Expected and actual previous trigger path not the same size" << std::endl;
          abort();
        }
      }

      // Look again using the TriggerNames class instead
      // of going to the service.

      TriggerNames const& triggerNamesFromEvent = e.triggerNames(*prod[0]);

      Strings namesFromEvent = triggerNamesFromEvent.triggerNames();
      if(namesFromEvent.size() != expected_trigger_previous_.size()) {
        std::cerr << "TestTriggerNames: While exercising TriggerNames class\n"
             << "Expected and actual previous trigger path lists not the same size" << std::endl;
        abort();
      }
      for(Strings::size_type i = 0; i < expected_trigger_previous_.size(); ++i) {
        if(namesFromEvent[i] != expected_trigger_previous_[i]) {
          std::cerr << "TestTriggerNames: While exercising TriggerNames class\n"
               << "Expected and actual previous trigger paths don't match" << std::endl;
          abort();
        }
        if(triggerNamesFromEvent.triggerName(i) != expected_trigger_previous_[i]) {
          std::cerr << "TestTriggerNames: While exercising TriggerNames class\n"
               << "name from index accessor\n"
               << "Expected and actual previous trigger paths don't match" << std::endl;
          abort();
        }
        // Exercise the function that returns an index
        if(i != triggerNamesFromEvent.triggerIndex(expected_trigger_previous_[i])) {
          std::cerr << "TestTriggerNames: While exercising TriggerNames class\n"
               << "index from name accessor\n"
               << "Expected and actual previous trigger paths don't match" << std::endl;
          abort();
        }
        if(triggerNamesFromEvent.size() != expected_trigger_previous_.size()) {
          std::cerr << "TestTriggerNames: While exercising TriggerNames class\n"
               << "Checking size accessor\n"
               << "Expected and actual previous trigger paths don't match" << std::endl;
          abort();
        }
      }
      // This causes it to find the results in the map lookup in the TEST configuration
      // and exercises that execution path in the code.
      // If you follow the execution in the debugger in EventBase::triggerNames_ in EventBase.cc
      // you can verify this is working correctly.
      if(prod.size() > 1U) {
        e.triggerNames(*prod[1]);
      }
    }

    edm::TriggerResultsByName resultsByNameHLT = e.triggerResultsByName("HLT");
    if(resultsByNameHLT.isValid() && expectedTriggerResultsHLT_.size() > 0) {

      edm::InputTag tag("TriggerResults", "", "HLT");
      edm::Handle<edm::TriggerResults> hTriggerResults;
      e.getByLabel(tag, hTriggerResults);

      if(hTriggerResults->parameterSetID() != resultsByNameHLT.parameterSetID() ||
           hTriggerResults->wasrun() != resultsByNameHLT.wasrun() ||
           hTriggerResults->accept() != resultsByNameHLT.accept() ||
           hTriggerResults->error() != resultsByNameHLT.error() ||
           hTriggerResults->at(0).state() != resultsByNameHLT.at("p01").state() ||
           hTriggerResults->at(0).index() != resultsByNameHLT.at("p01").index() ||
           (*hTriggerResults)[0].state() != resultsByNameHLT["p01"].state() ||
           (*hTriggerResults)[0].index() != resultsByNameHLT["p01"].index() ||
           hTriggerResults->wasrun(0) != resultsByNameHLT.wasrun("p01") ||
           hTriggerResults->accept(0) != resultsByNameHLT.accept("p01") ||
           hTriggerResults->error(0) != resultsByNameHLT.error("p01") ||
           hTriggerResults->state(0) != resultsByNameHLT.state("p01") ||
           hTriggerResults->index(0) != resultsByNameHLT.index("p01") ||
           hTriggerResults->at(0).state() != resultsByNameHLT.at(0).state() ||
           hTriggerResults->at(0).index() != resultsByNameHLT.at(0).index() ||
           (*hTriggerResults)[0].state() != resultsByNameHLT[0].state() ||
           (*hTriggerResults)[0].index() != resultsByNameHLT[0].index() ||
           hTriggerResults->wasrun(0) != resultsByNameHLT.wasrun(0) ||
           hTriggerResults->accept(0) != resultsByNameHLT.accept(0) ||
           hTriggerResults->error(0) != resultsByNameHLT.error(0) ||
           hTriggerResults->state(0) != resultsByNameHLT.state(0) ||
           hTriggerResults->index(0) != resultsByNameHLT.index(0)) {
        std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                  << "TriggerResults values do not match TriggerResultsByName values" << std::endl;
        abort();
      }
      edm::TriggerNames const& names = e.triggerNames(*hTriggerResults);
      if(names.triggerNames() != resultsByNameHLT.triggerNames() ||
          names.size() != resultsByNameHLT.size() ||
          names.triggerName(0) != resultsByNameHLT.triggerName(0) ||
          names.triggerIndex("p01") != resultsByNameHLT.triggerIndex("p01")) {
        std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                  << "TriggerNames values do not match TriggerResultsByName values" << std::endl;
        abort();
      }
    }

    if(expectedTriggerResultsHLT_.size() > iEvent_) {
      edm::TriggerResultsByName resultsByNameHLT = e.triggerResultsByName("HLT");
      if(!resultsByNameHLT.isValid()) {
        std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                  << "Invalid object for HLT" << std::endl;
        abort();
      }
      if(resultsByNameHLT.accept("p02") != (expectedTriggerResultsHLT_[iEvent_] == 1)) {
        std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                  << "Expected and actual HLT trigger results don't match" << std::endl;
        abort();
      }
      std::cout << "Event " << iEvent_ << "  " << resultsByNameHLT.accept("p02")
                << std::endl;
    }

    if(expectedTriggerResultsPROD_.size() > iEvent_) {
      edm::TriggerResultsByName resultsByNamePROD = e.triggerResultsByName("PROD");
      if(!resultsByNamePROD.isValid()) {
        std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                  << "Invalid object for PROD" << std::endl;
        abort();
      }
      if(resultsByNamePROD.accept("p1") != (expectedTriggerResultsPROD_[iEvent_] == 1)) {
        std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                  << "Expected and actual PROD trigger results don't match" << std::endl;
        abort();
      }
      std::cout << "Event " << iEvent_ << "  " << resultsByNamePROD.accept("p1") << std::endl;
    }

    edm::TriggerResultsByName resultsByNameNONE = e.triggerResultsByName("NONE");
    if(resultsByNameNONE.isValid()) {
      std::cerr << "TestTriggerNames: While testing TriggerResultsByName class\n"
                << "Object is valid with nonexistent process name" << std::endl;
      abort();
    }
    ++iEvent_;
  }

  void TestTriggerNames::endJob() {
  }
}

using edmtest::TestTriggerNames;

DEFINE_FWK_MODULE(TestTriggerNames);
