// Change Log
//
// 1 - M Fischler 2/8/08 Enable partial wildcards, as in HLT* or !CAL*
//			 A version of this code with cerr debugging traces has
//			 been placed in the doc area.  
// 			 See ../doc/EventSelector-behavior.doc for details of
//			 reactions to Ready or Exception states.


#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/algorithm/string.hpp"
#include "boost/regex.hpp"

#include <algorithm>

namespace edm
{
  EventSelector::EventSelector(Strings const& pathspecs,
			       Strings const& names):
    accept_all_(false),
    decision_bits_(),
    nonveto_bits_(),
    results_from_current_process_(true),
    psetID_initialized_(false),
    psetID_(),
    paths_(),
    nTriggerNames_(0),
    notStarPresent_(false)
  {
    init(pathspecs, names);
  }

  EventSelector::EventSelector(Strings const& pathspecs):
    accept_all_(false),
    decision_bits_(),
    nonveto_bits_(),
    results_from_current_process_(false),
    psetID_initialized_(false),
    psetID_(),
    paths_(pathspecs),
    nTriggerNames_(0),
    notStarPresent_(false)
  {
  }

  void
  EventSelector::init(Strings const& paths,
		      Strings const& triggernames)
  {
    // std::cerr << "### init entered\n";
    accept_all_ = false;
    decision_bits_.clear();

    nTriggerNames_ = triggernames.size();
    notStarPresent_ = false;

    if ( paths.empty() )
      {
	accept_all_ = true;
	return;
      }

    bool star_done = false;
    for (Strings::const_iterator i(paths.begin()), end(paths.end()); 
	 i!=end; ++i)
      {

	std::string current_path(*i);
	boost::erase_all(current_path, " \t");
	if (current_path == "!*")
	  {
            notStarPresent_ = true;
	  }
	if (current_path == "*")
	  {
            star_done = true;
	  }
	bool negative_criterion = false;
	if (current_path[0] == '!') {
	  negative_criterion = true;
	}
	std::string const& realname =
	      negative_criterion 
	      ? std::string((current_path.begin()+1), current_path.end())
	      : current_path;

	// instead of "see if the name can be found in the full list of paths"
	// we want to find all paths that match this name.	
	std::vector< Strings::const_iterator> matches =
		matching_triggers(triggernames, realname);
	
	if (matches.empty() && (glob2reg(realname) == realname)) 
	{
            throw edm::Exception(edm::errors::Configuration)
              << "EventSelector::init, An OutputModule is using SelectEvents\n"
                 "to request a trigger name that does not exist\n"
              << "The unknown trigger name is: " << realname << "\n";  
	}
		 	 
#ifdef REALNAME_MUST_MATCH_FEATURE_MIGHT_BE_WANTED
	if (matches.empty() && (glob2reg(realname) != realname)) 
	{
            throw edm::Exception(edm::errors::Configuration)
              << "EventSelector::init, An OutputModule is using SelectEvents\n"
                 "to request a wildcarded trigger name that does not match any trigger \n"
              << "The wildcarded trigger name is: " << realname << "\n";  
	}
#endif

	if ( !negative_criterion ) {
	  for (unsigned int t = 0; t != matches.size(); ++t) {
	    BitInfo bi(distance(triggernames.begin(),matches[t]), true);
	    decision_bits_.push_back(bi);
	  }
	} else if (negative_criterion) {
	  if (matches.empty()) {
              throw edm::Exception(edm::errors::Configuration)
              << "EventSelector::init, An OutputModule is using SelectEvents\n"
                 "to request a veto a set trigger names that do not exist\n"
              << "The problematic name is: " << current_path << "\n";  
	  
	  } else if (matches.size() == 1) {
	    BitInfo bi(distance(triggernames.begin(),matches[0]), false);
	    decision_bits_.push_back(bi);
	  } else {
	    Bits nonveto;
	    for (unsigned int t = 0; t != matches.size(); ++t) {
	      BitInfo bi(distance(triggernames.begin(),matches[t]), true);
	      // We set this to true because if the trigger bit is set,
	      // we want acceptTriggerBits to return true so that we will
	      // **reject** the criterion.
	      nonveto.push_back(bi);
	    }
	    nonveto_bits_.push_back(nonveto);
	  } 	
	}
    } // end of the for loop on i(paths.begin()), end(paths.end())

    if (notStarPresent_ && star_done) accept_all_ = true;

    // std::cerr << "### init exited\n";

  } // EventSelector::init
  
  EventSelector::EventSelector(edm::ParameterSet const& config,
			       Strings const& triggernames):
    accept_all_(false),
    decision_bits_(),
    results_from_current_process_(true),
    psetID_initialized_(false),
    psetID_(),
    paths_(),
    nTriggerNames_(0),
    notStarPresent_(false)
  {
    Strings paths; // default is empty...

    if (!config.empty())
      paths = config.getParameter<Strings>("SelectEvents");

    init(paths, triggernames);
  }


  bool EventSelector::acceptEvent(TriggerResults const& tr)
  {
    // For the current process we already initialized in the constructor,
    // The trigger names will not change so we can skip initialization.
    if (!results_from_current_process_) {
  
      // For previous processes we need to get the trigger names that
      // correspond to the bits in TriggerResults from the ParameterSet
      // set registry, which is stored once per file.  The ParameterSetID
      // stored in TriggerResults is the key used to find the info in the
      // registry.  We optimize using the fact the ID is unique. If the ID
      // has not changed since the last time we initialized with new triggernames,
      // then the names have not changed and we can skip this initialization.
      if ( !(psetID_initialized_ && psetID_ == tr.parameterSetID()) ) {

        Strings triggernames;
        bool fromPSetRegistry;

        edm::Service<edm::service::TriggerNamesService> tns;
        if (tns->getTrigPaths(tr, triggernames, fromPSetRegistry)) {

          init(paths_, triggernames);

          if (fromPSetRegistry) {
            psetID_ = tr.parameterSetID();
            psetID_initialized_ = true;
          }
          else {
            psetID_initialized_ = false;
          }
        }
        // This should never happen
        else {
          throw edm::Exception(edm::errors::Unknown)
            << "EventSelector::acceptEvent cannot find the trigger names for\n"
               "a process for which the configuration has requested that the\n"
               "OutputModule use TriggerResults to select events from.  This should\n"
               "be impossible, please send information to reproduce this problem to\n"
               "the edm developers.\n"; 
	}
      }
    }

    Bits::const_iterator i(decision_bits_.begin()),e(decision_bits_.end());
    for(;i!=e;++i)
      {
        if ( this->acceptTriggerPath(tr[i->pos_], *i) )
          {
            return true;
          }
      }
    
    // handle each entry in nonveto_bits_:
    // the previously special "!*" case is now handled routinely by this code.
    
    for (std::vector<Bits>::const_iterator nv =  nonveto_bits_.begin();
    					   nv != nonveto_bits_.end(); ++nv)
    {
      bool allFail = true;
      Bits::const_iterator i(nv->begin());
      Bits::const_iterator e(nv->end());
      for(;i!=e;++i) 
      {
        if (this->acceptTriggerPath(tr[i->pos_], *i)) 
	{ allFail = false; break; }
      }
      if (allFail) return true;
    }

    // If we have not accepted based on decision_bits_, nor on any one of
    // the nonveto_bits_ collections, then we reject this event.
    
    return false;
  }

  bool 
  EventSelector::acceptEvent(unsigned char const* array_of_trigger_results, 
  			     int number_of_trigger_paths) const
  {

    // This should never occur unless someone uses this function in
    // an incorrect way ...
    if (!results_from_current_process_) {
      throw edm::Exception(edm::errors::Configuration)
        << "\nEventSelector.cc::acceptEvent, you are attempting to\n"
        << "use a bit array for trigger results instead of the\n"
        << "TriggerResults object for a previous process.  This\n"
        << "will not work and ought to be impossible\n";
    }

    Bits::const_iterator i(decision_bits_.begin()),e(decision_bits_.end());
    for(;i!=e;++i)
      {
        int pathIndex = i->pos_;
        if (pathIndex < number_of_trigger_paths)
          {
            int byteIndex = ((int) pathIndex / 4);
            int subIndex = pathIndex % 4;
            int state = array_of_trigger_results[byteIndex] >> (subIndex * 2);
            state &= 0x3;
            HLTPathStatus pathStatus(static_cast<hlt::HLTState>(state));
            if ( this->acceptTriggerPath(pathStatus, *i) )
              {
                return true;
              }
          }
      }

    // handle each entry in nonveto_bits_:
    // the previously special "!*" case is now handled routinely by this code.
    
    for (std::vector<Bits>::const_iterator nv =  nonveto_bits_.begin();
    					   nv != nonveto_bits_.end(); ++nv)
    {
      bool allFail = true;
      Bits::const_iterator i(nv->begin());
      Bits::const_iterator e(nv->end());
      for(;i!=e;++i) 
      {
        int pathIndex = i->pos_;
        if (pathIndex < number_of_trigger_paths)
        {
            int byteIndex = ((int) pathIndex / 4);
            int subIndex = pathIndex % 4;
            int state = array_of_trigger_results[byteIndex] >> (subIndex * 2);
            state &= 0x3;
            HLTPathStatus pathStatus(static_cast<hlt::HLTState>(state));
 	    if ( this->acceptTriggerPath(pathStatus, *i) )
	    { allFail = false; break; }
        }
      }
      if (allFail) return true;
    }
    
    // If we have not accepted based on decision_bits_, nor on any one of
    // the nonveto_bits_ collections, then we reject this event.
    
    return false;

  }

  /**
   * Tests if the specified trigger selection list (path spec) is valid
   * in the context of the specified full trigger list.  Each element in
   * the selection list is tested to see if it possible for some
   * combination of trigger results to satisfy the selection.  If all
   * selection elements can be satisfied one way or another, then this
   * method returns true.  If one or more selection elements could never
   * be satisfied given the input full trigger list, then this method
   * returns false.  At some level, this method tests whether the selection
   * list is a "subset" of the full trigger list.
   *
   * @param pathspec The trigger selection list (vector of string).
   * @param fullTriggerList The full list of trigger names (vector of string).
   * @return true if the selection list is valid, false otherwise.
   */
  bool EventSelector::selectionIsValid(Strings const& pathspec,
                                       Strings const& fullTriggerList)
  {
    // an empty selection list is not valid
    // (we default an empty "SelectEvents" parameter to {"*","!*"} in
    // the getEventSelectionVString method below to help avoid this)
    if (pathspec.size() == 0)
    {
      return false;
    }

    // loop over each element in the selection list
    for (unsigned int idx = 0; idx < pathspec.size(); idx++)
    {
      Strings workingList;
      workingList.push_back(pathspec[idx]);

      // catch exceptions from the EventSelector constructor
      // (and anywhere else) and mark those as failures.
      // The EventSelector constructor seems to do the work of
      // checking if the selection is outside the full trigger list.
      try
      {
        // create an EventSelector instance for this selection
        EventSelector evtSelector(workingList, fullTriggerList);

        // create the TriggerResults instance that we'll use for testing
        unsigned int fullTriggerCount = fullTriggerList.size();
        HLTGlobalStatus hltGS(fullTriggerCount);
        TriggerResults sampleResults(hltGS, fullTriggerList);

        // loop over each path
        bool oneResultMatched = false;
        for (unsigned int iPath = 0; iPath < fullTriggerCount; iPath++)
        {
          // loop over the possible values for the path status
          for (int iState = static_cast<int>(hlt::Pass);
               iState <= static_cast<int>(hlt::Exception);
               iState++)
          {
            sampleResults[iPath] = HLTPathStatus(static_cast<hlt::HLTState>(iState), 0);
            if (evtSelector.wantAll() || evtSelector.acceptEvent(sampleResults))
            {
              oneResultMatched = true;
              break;
            }

            sampleResults.reset(iPath);
          }

          if (oneResultMatched) break;
        }

        // if none of the possible trigger results matched the
        // selection element, then we declare the whole selection
        // list invalid
        if (! oneResultMatched)
        {
          return false;
        }
      }
      catch (const edm::Exception& excpt)
      {
        return false;
      }
    }

    // if we made it to this point, then it must have been possible
    // to satisfy every selection element one way or another
    return true;
  }

  /**
   * Tests if the specified trigger selection lists (path specs) overlap,
   * where "overlap" means that a valid trigger result (given the full
   * trigger list) could satisfy both selections.
   *
   * @param pathspec1 The first trigger selection list (vector of string).
   * @param pathspec2 The second trigger selection list (vector of string).
   * @param fullTriggerList The full list of trigger names (vector of string).
   * @return OverlapResult which indicates the degree of overlap.
   */
  evtSel::OverlapResult
  EventSelector::testSelectionOverlap(Strings const& pathspec1,
                                      Strings const& pathspec2,
                                      Strings const& fullTriggerList)
  {
    // first, test that the selection lists are valid
    if (! selectionIsValid(pathspec1, fullTriggerList) ||
        ! selectionIsValid(pathspec2, fullTriggerList))
    {
      return evtSel::InvalidSelection;
    }

    // initialize possible states
    bool noOverlap = true;
    bool exactMatch = true;

    // catch exceptions from the EventSelector constructor
    // (and anywhere else) and mark those as failures
    try
    {
      // create an EventSelector instance for each selection list
      EventSelector selector1(pathspec1, fullTriggerList);
      EventSelector selector2(pathspec2, fullTriggerList);

      // create the TriggerResults instance that we'll use for testing
      unsigned int fullTriggerCount = fullTriggerList.size();
      HLTGlobalStatus hltGS(fullTriggerCount);
      TriggerResults sampleResults(hltGS, fullTriggerList);

      // loop over each path
      for (unsigned int iPath = 0; iPath < fullTriggerCount; iPath++)
      {
        // loop over the possible values for the path status
        for (int iState = static_cast<int>(hlt::Pass);
             iState <= static_cast<int>(hlt::Exception);
             iState++)
        {
          sampleResults[iPath] =
            HLTPathStatus(static_cast<hlt::HLTState>(iState), 0);
          bool accept1 = selector1.wantAll() ||
            selector1.acceptEvent(sampleResults);
          bool accept2 = selector2.wantAll() ||
            selector2.acceptEvent(sampleResults);
          if (accept1 != accept2)
          {
            exactMatch = false;
          }
          if (accept1 && accept2)
          {
            noOverlap = false;
          }
          sampleResults.reset(iPath);
        }
      }
    }
    catch (const edm::Exception& excpt)
    {
      return evtSel::InvalidSelection;
    }

    if (exactMatch) {return evtSel::ExactMatch;}
    if (noOverlap) {return evtSel::NoOverlap;}
    return evtSel::PartialOverlap;
  }

  /**
   * Applies a trigger selection mask to a specified trigger result object.
   * Within the trigger result object, each path status is left unchanged
   * if it satisfies the trigger selection (path specs) or cleared if it
   * does not satisfy the trigger selection.  In this way, the resulting
   * trigger result object contains only path status values that "pass"
   * the selection criteria.
   *
   * @param pathspecs The trigger selection list (vector of string).
   * @param inputResults The raw trigger results object that will be masked.
   * @param fullTriggerList The full list of trigger names (vector of string).
   * @return a copy of the input trigger results object with only the path
   *         status results that match the trigger selection.
   * @throws edm::Exception if the number of paths in the TriggerResults
   *         object does not match the specified full trigger list, or
   *         if the trigger selection is invalid in the context of the
   *         full trigger list.
   */
  boost::shared_ptr<TriggerResults>
  EventSelector::maskTriggerResults(Strings const& pathspecs,
                                    TriggerResults const& inputResults,
                                    Strings const& fullTriggerList)
  {
    // fetch and validate the total number of paths
    unsigned int fullTriggerCount = fullTriggerList.size();
    if (fullTriggerCount != inputResults.size())
    {
      throw edm::Exception(edm::errors::EventCorruption)
        << "EventSelector::maskTriggerResults, the TriggerResults\n"
        << "size (" << inputResults.size()
        << ") does not match the number of paths in the\n"
        << "full trigger list (" << fullTriggerCount << ").\n";
    }

    // create a working copy of the TriggerResults object
    HLTGlobalStatus hltGS(fullTriggerCount);
    boost::shared_ptr<TriggerResults>
      maskedResults(new TriggerResults(hltGS, inputResults.parameterSetID()));
    for (unsigned int iPath = 0; iPath < fullTriggerCount; iPath++)
    {
      (*maskedResults)[iPath] = inputResults[iPath];
    }

    // create an EventSelector to use when testing if a path status passes
    EventSelector selector(pathspecs, fullTriggerList);

    // create the TriggerResults instance that we'll use for testing
    HLTGlobalStatus hltGS2(fullTriggerCount);
    TriggerResults sampleResults(hltGS2, fullTriggerList);

    // loop over each path and reset the path status if needed
    for (unsigned int iPath = 0; iPath < fullTriggerCount; iPath++)
    {
      sampleResults[iPath] = (*maskedResults)[iPath];
      if (! selector.wantAll() && ! selector.acceptEvent(sampleResults))
      {
        maskedResults->reset(iPath);
      }
      sampleResults.reset(iPath);
    }
    return maskedResults;
  }

  /**
   * Returns the list of strings that correspond to the trigger
   * selection request in the specified parameter set (the list
   * of strings contained in the "SelectEvents" parameter).
   *
   * @param pset The ParameterSet that contains the trigger selection.
   * @return the trigger selection list (vector of string).
   */
  std::vector<std::string>
  EventSelector::getEventSelectionVString(edm::ParameterSet const& pset)
  {
    // default the selection to everything (wildcard)
    Strings selection;
    selection.push_back("*");
    selection.push_back("!*");

    // the SelectEvents parameter is a ParameterSet within
    // a ParameterSet, so we have to pull it out twice
    ParameterSet selectEventsParamSet =
      pset.getUntrackedParameter("SelectEvents", ParameterSet());
    if (! selectEventsParamSet.empty()) {
      Strings path_specs = 
        selectEventsParamSet.getParameter<Strings>("SelectEvents");
      if (! path_specs.empty()) {
        selection = path_specs;
      }
    }

    // return the result
    return selection;
  }

  bool EventSelector::acceptTriggerPath(HLTPathStatus const& pathStatus,
                                        BitInfo const& pathInfo) const
  {
    return ( ((pathStatus.state()==hlt::Pass) &&  (pathInfo.accept_state_)) ||
             ((pathStatus.state()==hlt::Fail) && !(pathInfo.accept_state_)) ||
             ((pathStatus.state()==hlt::Exception)) );
  }

  std::string  EventSelector::glob2reg(std::string const& s) 
  {
    std::string r = s;
    boost::replace_all(r, "*", ".*");
    boost::replace_all(r, "?", ".");
    return r;
  }

  std::vector< EventSelector::Strings::const_iterator > 
  EventSelector::matching_triggers(Strings const& trigs, std::string const& s) 
  {
    std::vector< Strings::const_iterator > m;
    boost::regex r ( glob2reg(s) );
    for (Strings::const_iterator i = trigs.begin(); i != trigs.end(); ++i)
    {
      if  (boost::regex_match((*i),r)) 
      {
        m.push_back(i);
      }
    }
    return m;
  }

}
