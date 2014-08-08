#include <algorithm>

#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"

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
                                std::vector<std::string> const& iAllTriggerNames,
                                edm::detail::TriggerResultsBasedEventSelector& oSelector) {
      // If selectevents is an emtpy ParameterSet, then we are to write
      // all events, or one which contains a vstrig 'SelectEvents' that
      // is empty, we are to write all events. We have no need for any
      // EventSelectors.
      if(iPSet.empty()) {
        oSelector.setupDefault(iAllTriggerNames);
        return true;
      }
      
      std::vector<std::string> path_specs =
      iPSet.getParameter<std::vector<std::string> >("SelectEvents");
      
      if(path_specs.empty()) {
        oSelector.setupDefault(iAllTriggerNames);
        return true;
      }
      
      // If we get here, we have the possibility of having to deal with
      // path_specs that look at more than one process.
      std::vector<parsed_path_spec_t> parsed_paths(path_specs.size());
      for(size_t i = 0; i < path_specs.size(); ++i) {
        parse_path_spec(path_specs[i], parsed_paths[i]);
      }
      oSelector.setup(parsed_paths, iAllTriggerNames, iProcessName);
      
      return false;
    }

    
    void NamedEventSelector::fill(EventPrincipal const& e, ModuleCallingContext const* mcc) {
      edm::BasicHandle h = e.getByLabel(PRODUCT_TYPE,
                                        s_TrigResultsType,
                                        inputTag_,
                                        nullptr,
                                        mcc);
      convert_handle(std::move(h),product_);
    }
    
    typedef detail::NamedEventSelector NES;


    TriggerResultsBasedEventSelector::TriggerResultsBasedEventSelector() :
      fillDone_(false),
      numberFound_(0),
      selectors_()
    { }

    void
    TriggerResultsBasedEventSelector::setupDefault(std::vector<std::string> const& triggernames)
    {

      // Set up one NamedEventSelector, with default configuration
      std::vector<std::string> paths;
      EventSelector es(paths, triggernames);
      selectors_.emplace_back("", es);
      //selectors_.push_back(NES("", EventSelector("",triggernames)));
    }

    void
    TriggerResultsBasedEventSelector::setup(std::vector<parsed_path_spec_t> const& path_specs,
			  std::vector<std::string> const& triggernames,
                          const std::string& process_name)
    {
      // paths_for_process maps each PROCESS names to a sequence of
      // PATH names
      std::map<std::string, std::vector<std::string> > paths_for_process;
      for (std::vector<parsed_path_spec_t>::const_iterator 
	     i = path_specs.begin(), 
	     e = path_specs.end();
 	   i != e;
	   ++i)
	{
          // Default to current process if none specified
          if (i->second == "") {
            paths_for_process[process_name].push_back(i->first);
          }
          else {
            paths_for_process[i->second].push_back(i->first);
          }
	}
      // Now go through all the PROCESS names, and create a
      // NamedEventSelector for each.
      for (std::map<std::string, std::vector<std::string> >::const_iterator
	     i = paths_for_process.begin(),
	     e = paths_for_process.end();
	   i != e;
	   ++i)
	{
          // For the current process we know the trigger names
          // from the configuration file
          if (i->first == process_name) {
            selectors_.emplace_back(i->first, 
				    EventSelector(i->second, 
				    triggernames));
          }
          // For previous processes we do not know the trigger
          // names yet.
          else {
            selectors_.emplace_back(i->first, EventSelector(i->second));
          }
	}
    }

    TriggerResultsBasedEventSelector::handle_t
    TriggerResultsBasedEventSelector::getOneTriggerResults(EventPrincipal const& ev, ModuleCallingContext const* mcc)
    {
      fill(ev, mcc);
      return returnOneHandleOrThrow();
    }

    bool
    TriggerResultsBasedEventSelector::wantEvent(EventPrincipal const& ev, ModuleCallingContext const* mcc)
    {
      // We have to get all the TriggerResults objects before we test
      // any for a match, because we have to deal with the possibility
      // of multiple TriggerResults objects --- note that the presence
      // of more than one TriggerResults object in the event is
      // intended to lead to an exception throw *unless* either the
      // configuration has been set to match all events, or the
      // configuration is set to use specific process names.

      fill(ev, mcc);

      // Now we go through and see if anyone matches...
      iter i = selectors_.begin();
      iter e = selectors_.end();
      bool match_found = false;
      while (!match_found && (i!=e))
	{
	  match_found = i->match();
	  ++i;
	}
      return match_found;
    }
    
    TriggerResultsBasedEventSelector::handle_t
    TriggerResultsBasedEventSelector::returnOneHandleOrThrow()
    {
      switch (numberFound_)
	{
	case 0:
	  throw edm::Exception(edm::errors::ProductNotFound,
			       "TooFewProducts")
	    << "TriggerResultsBasedEventSelector::returnOneHandleOrThrow: "
	    << " too few products found, "
	    << "exepcted one, got zero\n";
	case 1:

	  break;
	default:
	  throw edm::Exception(edm::errors::ProductNotFound,
			       "TooManyMatches")
	    << "TriggerResultsBasedEventSelector::returnOneHandleOrThrow: "
	    << "too many products found, "
	    << "expected one, got " << numberFound_ << '\n';
	}
      return selectors_[0].product();
    }

    TriggerResultsBasedEventSelector::size_type
    TriggerResultsBasedEventSelector::fill(EventPrincipal const& ev, ModuleCallingContext const* mcc)
    {
      if (!fillDone_)
	{
	  fillDone_ = true;
	  for (iter i = selectors_.begin(), e = selectors_.end(); 
	       i != e; ++i)
	    {
	      i->fill(ev, mcc);     // fill might throw...
	      ++numberFound_ ; // so numberFound_ might be less than expected
	    }
	}
      return numberFound_;
    }

    void
    TriggerResultsBasedEventSelector::clear()
    { 
      for_all(selectors_, std::bind(&NamedEventSelector::clear, std::placeholders::_1));
      fillDone_ = false;
      numberFound_ = 0;
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
        for (std::vector<std::pair<std::string, int> >::const_iterator i = iter->second.begin(), e = iter->second.end();
             i != e; ++i) {
          endPaths.push_back(i->first);
          endPathPositions.push_back(i->second);
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
