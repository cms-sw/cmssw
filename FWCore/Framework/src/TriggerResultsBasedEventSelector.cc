#include <algorithm>

#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

static const edm::TypeID s_TrigResultsType(typeid(edm::TriggerResults));

namespace {
  //--------------------------------------------------------
  // Remove whitespace (spaces and tabs) from a std::string.
  void remove_whitespace(std::string& s) {
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\t'), s.end());
  }

  void test_remove_whitespace() {
    std::string a("noblanks");
    std::string b("\t   no   blanks    \t");

    remove_whitespace(b);
    assert(a == b);
  }

  //--------------------------------------------------------
  // Given a path-spec (std::string of the form "a:b", where the ":b" is
  // optional), return a parsed_path_spec_t containing "a" and "b".

  typedef std::pair<std::string, std::string> parsed_path_spec_t;
  void parse_path_spec(std::string const& path_spec,
                       parsed_path_spec_t& output) {
    std::string trimmed_path_spec(path_spec);
    remove_whitespace(trimmed_path_spec);

    std::string::size_type colon = trimmed_path_spec.find(":");
    if(colon == std::string::npos) {
      output.first = trimmed_path_spec;
    } else {
      output.first  = trimmed_path_spec.substr(0, colon);
      output.second = trimmed_path_spec.substr(colon + 1,
                                               trimmed_path_spec.size());
    }
  }

  void test_parse_path_spec() {
    std::vector<std::string> paths;
    paths.push_back("a:p1");
    paths.push_back("b:p2");
    paths.push_back("  c");
    paths.push_back("ddd\t:p3");
    paths.push_back("eee:  p4  ");

    std::vector<parsed_path_spec_t> parsed(paths.size());
    for(size_t i = 0; i < paths.size(); ++i) {
      parse_path_spec(paths[i], parsed[i]);
    }

    assert(parsed[0].first  == "a");
    assert(parsed[0].second == "p1");
    assert(parsed[1].first  == "b");
    assert(parsed[1].second == "p2");
    assert(parsed[2].first  == "c");
    assert(parsed[2].second == "");
    assert(parsed[3].first  == "ddd");
    assert(parsed[3].second == "p3");
    assert(parsed[4].first  == "eee");
    assert(parsed[4].second == "p4");
  }
}

namespace edm
{
  namespace test {
    void run_all_output_module_tests() {
      test_remove_whitespace();
      test_parse_path_spec();
    }
  }

  namespace detail
  {

    bool configureEventSelector(edm::ParameterSet const& iPSet,
                                std::string const& iProcessName,
                                std::vector<std::string> const& iAllPathNames,
                                edm::detail::TriggerResultsBasedEventSelector& oSelector) {
      // If selectevents is an empty ParameterSet, then we are to write
      // all events, or one which contains a vstrig 'SelectEvents' that
      // is empty, we are to write all events. We have no need for any
      // EventSelectors.
      if(iPSet.empty()) {
        oSelector.setupDefault(iAllPathNames);
        return true;
      }

      std::vector<std::string> pathSpecs = iPSet.getParameter<std::vector<std::string> >("SelectEvents");

      if(pathSpecs.empty()) {
        oSelector.setupDefault(iAllPathNames);
        return true;
      }

      // If we get here, we have the possibility of having to deal with
      // pathSpecs that look at more than one process.
      std::vector<parsed_path_spec_t> parsedPathSpecs(pathSpecs.size());
      for(size_t i = 0; i < pathSpecs.size(); ++i) {
        parse_path_spec(pathSpecs[i], parsedPathSpecs[i]);
      }
      oSelector.setup(parsedPathSpecs, iAllPathNames, iProcessName);

      return false;
    }

    // The path names for prior processes may be different in different runs.
    // Check for this, and modify the selector accordingly if necessary.
    void
    NamedEventSelector::beginRun(ProcessHistory const& ph) {
      if(!eventSelector_.forCurrentProcess()) {
        ProcessConfiguration pc;
        bool found = ph.getConfigurationForProcess(inputTag_.process(), pc);
        if(!found) {
          throw edm::Exception(errors::Configuration) <<
            "Process name '" << inputTag_.process() <<
            "' is specified in 'SelectEvents', but does not appear in the process history.\n";
        }
        ParameterSetID const& psetID  = pc.parameterSetID();
        ParameterSet const* processParameterSet = pset::Registry::instance()->getMapped(psetID);
        ParameterSet const& triggerPathsPSet = processParameterSet->getParameterSet("@trigger_paths");
        ParameterSetID triggerPathsPSetID = triggerPathsPSet.id();
        eventSelector_.beginRun(triggerPathsPSetID);
      }
    }

    TriggerResultsBasedEventSelector::TriggerResultsBasedEventSelector() :
      selectors_()
    { }

    void
    TriggerResultsBasedEventSelector::setupDefault(std::vector<std::string> const& pathNames) {
      // Set up one NamedEventSelector, with default configuration
      std::vector<std::string> pathSpecs;
      EventSelector es(pathSpecs, pathNames);
      selectors_.emplace_back("", es);
    }

    void
    TriggerResultsBasedEventSelector::setup(std::vector<parsed_path_spec_t> const& pathSpecs,
                          std::vector<std::string> const& pathNames,
                          const std::string& processName) {
      // pathsForProcess maps each PROCESS names to a sequence of
      // PATH names
      std::map<std::string, std::vector<std::string> > pathsForProcess;
      for (auto const& pathSpec : pathSpecs) {
        // Default to current process if none specified
        if (pathSpec.second == "") {
          pathsForProcess[processName].push_back(pathSpec.first);
        }
        else {
          pathsForProcess[pathSpec.second].push_back(pathSpec.first);
        }
      }
      // Now go through all the PROCESS names, and create a
      // NamedEventSelector for each.
      for (auto const& pathForProcess : pathsForProcess) {
        // For the current process we know the trigger names
        // from the configuration file
        if (pathForProcess.first == processName) {
          selectors_.emplace_back(pathForProcess.first, EventSelector(pathForProcess.second, pathNames));
        } else {
          // For previous processes we do not know the trigger
          // names yet.
          selectors_.emplace_back(pathForProcess.first, EventSelector(pathForProcess.second));
        }
      }
    }

    // The path names for prior processes may be different in different runs.
    // Check for this, and modify each selectors accordingly when needed
    void
    TriggerResultsBasedEventSelector::beginRun(RunPrincipal const& rp) {
      ProcessHistory const& ph = rp.processHistory();
      for(auto& selector : selectors_) {
        selector.beginRun(ph);
      }
    }

    bool
    TriggerResultsBasedEventSelector::wantEvent(EventPrincipal const& ev, ModuleCallingContext const* mcc) const {
      for(auto& selector : selectors_) {
        edm::BasicHandle h = ev.getByLabel(PRODUCT_TYPE,
                                          s_TrigResultsType,
                                          selector.inputTag(),
                                          nullptr,
                                          mcc);
        handle_t product;
        convert_handle(std::move(h), product);
        bool match = selector.match(*product);
        if(match) {
          return true;
        }
      }
      return false;
    }

    ParameterSetID
    registerProperSelectionInfo(edm::ParameterSet const& iInitial,
                                std::string const& iLabel,
                                std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                bool anyProductProduced) {
      ParameterSet selectEventsInfo;
      selectEventsInfo.copyForModify(iInitial);
      selectEventsInfo.addParameter<bool>("InProcessHistory", anyProductProduced);
      std::vector<std::string> endPaths;
      std::vector<int> endPathPositions;

      // The label will be empty if and only if this is a SubProcess
      // SubProcess's do not appear on any end path
      if (!iLabel.empty()) {
        std::map<std::string, std::vector<std::pair<std::string, int> > >::const_iterator iter = outputModulePathPositions.find(iLabel);
        assert(iter != outputModulePathPositions.end());
        for(auto const& item : iter->second) {
          endPaths.push_back(item.first);
          endPathPositions.push_back(item.second);
        }
      }
      selectEventsInfo.addParameter<std::vector<std::string> >("EndPaths", endPaths);
      selectEventsInfo.addParameter<std::vector<int> >("EndPathPositions", endPathPositions);
      if (!selectEventsInfo.exists("SelectEvents")) {
        selectEventsInfo.addParameter<std::vector<std::string> >("SelectEvents", std::vector<std::string>());
      }
      selectEventsInfo.registerIt();

      return selectEventsInfo.id();
    }

  } // namespace detail
} // namespace edm
