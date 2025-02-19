
/*----------------------------------------------------------------------

 EDAnalyzer for testing EventSelectionID class and tracking mechanism.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <iterator>


namespace edm {
  class EventSetup;
}

namespace edmtest {

  class HistoryAnalyzer : public edm::EDAnalyzer {
  public:

    explicit HistoryAnalyzer(edm::ParameterSet const& params);
    void analyze(edm::Event const& event, edm::EventSetup const&);
    void endJob();

  private:

    typedef std::vector<std::string> vstring;

    int expectedSize_;
    int eventCount_;
    int expectedCount_;

    std::vector<edm::ParameterSet> expectedSelectEventsInfo_;
    vstring expectedPaths_;
    vstring expectedEndPaths_;
    vstring expectedModules_;
    vstring expectedDroppedEndPaths_;
    vstring expectedDroppedModules_;
    vstring expectedDropFromProcPSet_;
    edm::ParameterSet expectedModulesOnEndPaths_;
 };

  HistoryAnalyzer::HistoryAnalyzer(edm::ParameterSet const& params) :
    expectedSize_(params.getParameter<int>("expectedSize")),
    eventCount_(0),
    expectedCount_(params.getParameter<int>("expectedCount")),
    expectedSelectEventsInfo_(params.getParameter<std::vector<edm::ParameterSet> >("expectedSelectEventsInfo")),
    expectedPaths_(params.getParameter<vstring>("expectedPaths")),
    expectedEndPaths_(params.getParameter<vstring>("expectedEndPaths")),
    expectedModules_(params.getParameter<vstring>("expectedModules")),
    expectedDroppedEndPaths_(params.getParameter<vstring>("expectedDroppedEndPaths")),
    expectedDroppedModules_(params.getParameter<vstring>("expectedDroppedModules")),
    expectedDropFromProcPSet_(params.getParameter<vstring>("expectedDropFromProcPSet")),
    expectedModulesOnEndPaths_(params.getParameter<edm::ParameterSet>("expectedModulesOnEndPaths")) {
  }

  void
  HistoryAnalyzer::analyze(edm::Event const& event, edm::EventSetup const&)
  {
    edm::EventSelectionIDVector const& esv = event.eventSelectionIDs();
    assert(esv.size() == static_cast<size_t>(expectedSize_));

    for (unsigned i = 0; i < esv.size(); ++i) {
      edm::ParameterSet selectEventsInfo = getParameterSet(esv[i]);
      if (eventCount_ == 0) {
        std::cout << selectEventsInfo << std::endl;
      }
      if (i < expectedSelectEventsInfo_.size()) {
        assert(selectEventsInfo == expectedSelectEventsInfo_[i]);
      }
    }

    if (eventCount_ == 0) {
      edm::ParameterSet const& proc_pset = edm::getProcessParameterSet();

      edm::pset::Registry* reg = edm::pset::Registry::instance();

      if (!expectedPaths_.empty()) {
        vstring paths = proc_pset.getParameter<vstring>("@paths");
        assert(paths == expectedPaths_);

        for (vstring::const_iterator i = paths.begin(), iEnd = paths.end();
             i != iEnd; ++i) {
          vstring modulesOnPath = proc_pset.getParameter<std::vector<std::string> >(*i);
          assert(!modulesOnPath.empty());
        }
      }

      if (!expectedEndPaths_.empty()) {
        vstring end_paths = proc_pset.getParameter<vstring>("@end_paths");
        assert(end_paths == expectedEndPaths_);

        for (vstring::const_iterator i = end_paths.begin(), iEnd = end_paths.end();
             i != iEnd; ++i) {
          vstring modulesOnEndPath = proc_pset.getParameter<vstring>(*i);
          assert(!modulesOnEndPath.empty());
          vstring expectedModulesOnEndPath = expectedModulesOnEndPaths_.getParameter<vstring>(*i);
          if (expectedModulesOnEndPath != modulesOnEndPath) {
            std::copy(expectedModulesOnEndPath.begin(), expectedModulesOnEndPath.end(), std::ostream_iterator<std::string>(std::cout, " "));
            std::cout << std::endl;
            std::copy(modulesOnEndPath.begin(), modulesOnEndPath.end(), std::ostream_iterator<std::string>(std::cout, " "));
            std::cout << std::endl;
            assert(expectedModulesOnEndPath == modulesOnEndPath);
          }
        }
      }

      if (!expectedModules_.empty()) {
        vstring all_modules = proc_pset.getParameter<vstring>("@all_modules");

        if (all_modules != expectedModules_) {
          std::copy(all_modules.begin(), all_modules.end(), std::ostream_iterator<std::string>(std::cout, " "));
          std::cout << std::endl;
          std::copy(expectedModules_.begin(), expectedModules_.end(), std::ostream_iterator<std::string>(std::cout, " "));
          std::cout << std::endl;
          assert(all_modules == expectedModules_);
        }
        for (vstring::const_iterator i = all_modules.begin(), iEnd = all_modules.end();
             i != iEnd; ++i) {
          // Make sure the ParameterSet for the module is also present
          edm::ParameterSet const& pset = proc_pset.getParameterSet(*i);
          // This is probably overkill, but also check it can be retrieved by ID from the registry 
          edm::ParameterSetID id = pset.id();
	  edm::ParameterSet const* result = reg->getMapped(id);
          assert(result != 0);
        }
      }

      for (vstring::const_iterator i = expectedDroppedEndPaths_.begin(), iEnd = expectedDroppedEndPaths_.end();
           i != iEnd; ++i) {
        assert(!proc_pset.exists(*i));
      }

      for (vstring::const_iterator i = expectedDroppedModules_.begin(), iEnd = expectedDroppedModules_.end();
           i != iEnd; ++i) {
        assert(!proc_pset.exists(*i));

        for (edm::pset::Registry::const_iterator j = reg->begin(), jEnd = reg->end();
             j != jEnd; ++j) {
          if (j->second.exists("@module_label")) {
	    assert(j->second.getParameter<std::string>("@module_label") != *i);
	  }
        }
      }
      for (vstring::const_iterator i = expectedDropFromProcPSet_.begin(), iEnd = expectedDropFromProcPSet_.end();
           i != iEnd; ++i) {
        assert(!proc_pset.existsAs<edm::ParameterSet>(*i,true));
        assert(proc_pset.existsAs<edm::ParameterSet>(*i,false));
        bool isInRegistry = false;
        for (edm::pset::Registry::const_iterator j = reg->begin(), jEnd = reg->end();
             j != jEnd; ++j) {
          if (j->second.exists("@module_label")) {
            if (j->second.getParameter<std::string>("@module_label") == *i) isInRegistry = true;
	  }
        }
        assert(isInRegistry);
      }
    }
    ++eventCount_;
  }

  void
  HistoryAnalyzer::endJob()
  {
    // std::cout << "Expected count is: " << expectedCount_ << std::endl;
    // std::cout << "Event count is:    " << eventCount_ << std::endl;
    assert(eventCount_ == expectedCount_);
  }
}

using edmtest::HistoryAnalyzer;
DEFINE_FWK_MODULE(HistoryAnalyzer);
