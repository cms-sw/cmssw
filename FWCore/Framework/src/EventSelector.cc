// Change Log
//
// 1 - M Fischler 2/8/08 Enable partial wildcards, as in HLT* or !CAL*
//			 A version of this code with cerr debugging traces has
//			 been placed in the doc area.  
// 			 See ../doc/EventSelector-behavior.doc for details of
//			 reactions to Ready or Exception states.
// 1a M Fischler 2/13/08 Clear the all_must_fail_ array at the start of init.
//			 This is needed in the case of paths with wildcards,
//			 on explicit processes other than a current process
//			 (in which case init() is called whenever the trigger
//			 PSetID changes, and we don't want the old array
//			 contents to stick around.
//
// 2 - M Fischler 2/21/08 (In preparation for "exception-awareness" features):
//			 Factored out the decision making logic from the 
//			 two forms of acceptEvent, into the single routine
//			 selectionDecision().
//
// 3 - M Fischler 2/25/08 (Toward commit of "exception-awareness" features):
//			 @exception and noexception& features 
//
// 4- M Fischler 2/28/08 Repair ommision in selectionIsValid when pathspecs
//			is just "!*"
//
// 5- M Fischler 3/3/08 testSelectionOverlap and maskTriggerResults appropriate
//			for the new forms of pathspecs
//
// 6 - K Biery 03/24/08 modified maskTriggerResults (no longer static) to
//                      avoid performance penalty of creating a new
//                      EventSelector instance for each call (in static case)
//


#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/algorithm/string.hpp"
#include "boost/regex.hpp"

#include <algorithm>
#include <cassert>

namespace edm
{
  EventSelector::EventSelector(Strings const& pathspecs,
			       Strings const& names):
    accept_all_(false),
    absolute_acceptors_(),
    conditional_acceptors_(),
    exception_acceptors_(),
    all_must_fail_(),
    all_must_fail_noex_(),
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
    absolute_acceptors_(),
    conditional_acceptors_(),
    exception_acceptors_(),
    all_must_fail_(),
    all_must_fail_noex_(),
    results_from_current_process_(false),
    psetID_initialized_(false),
    psetID_(),
    paths_(pathspecs),
    nTriggerNames_(0),
    notStarPresent_(false)
  {
  }

  EventSelector::EventSelector(ParameterSet const& config,
			       Strings const& triggernames):
    accept_all_(false),
    absolute_acceptors_(),
    conditional_acceptors_(),
    exception_acceptors_(),
    all_must_fail_(),
    all_must_fail_noex_(),
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

  void
  EventSelector::init(Strings const& paths,
		      Strings const& triggernames)
  {
    // std::cerr << "### init entered\n";
    accept_all_ = false;
    absolute_acceptors_.clear(),
    conditional_acceptors_.clear(),
    exception_acceptors_.clear(),
    all_must_fail_.clear();
    all_must_fail_noex_.clear();
    nTriggerNames_ = triggernames.size();
    notStarPresent_ = false;

    if (paths.empty())
      {
	accept_all_ = true;
	return;
      }

    // The following are for the purpose of establishing accept_all_ by 
    // virtue of an inclusive set of paths:
    bool unrestricted_star = false;
    bool negated_star      = false;
    bool exception_star    = false;
    
    for (Strings::const_iterator i(paths.begin()), end(paths.end()); 
	 i!=end; ++i)
    {
      std::string pathSpecifier(*i);
      boost::erase_all(pathSpecifier, " \t"); // whitespace eliminated
      if (pathSpecifier == "*")           unrestricted_star = true;
      if (pathSpecifier == "!*")          negated_star = true;
      if (pathSpecifier == "exception@*") exception_star = true;

      std::string basePathSpec(pathSpecifier);
      bool noex_demanded = false;
      std::string::size_type 
	      and_noexception = pathSpecifier.find("&noexception");
      if (and_noexception != std::string::npos) {
	basePathSpec = pathSpecifier.substr(0,and_noexception);
        noex_demanded = true;
      }
      std::string::size_type and_noex = pathSpecifier.find("&noex");
      if (and_noex != std::string::npos) {
	basePathSpec = pathSpecifier.substr(0,and_noexception);
        noex_demanded = true;
      }
      and_noexception = basePathSpec.find("&noexception");
      and_noex = basePathSpec.find("&noex");	
      if (and_noexception != std::string::npos ||
	   and_noex != std::string::npos)
          throw edm::Exception(errors::Configuration)
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request a trigger name, but specifying &noexceptions twice\n"
            << "The improper trigger name is: " << pathSpecifier << "\n";  

      std::string realname(basePathSpec);
      bool negative_criterion = false;
      if (basePathSpec[0] == '!') {
	negative_criterion = true;
	realname = basePathSpec.substr(1,std::string::npos);
      }
      bool exception_spec = false;
      if (realname.find("exception@") == 0) {
	exception_spec = true;  
	realname = realname.substr(10, std::string::npos); 
	// strip off 10 chars, which is length of "exception@" 
      }	
      if (negative_criterion &&  exception_spec)
          throw edm::Exception(errors::Configuration)
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request a trigger name starting with !exception@.\n"
	       "This is not supported.\n"
            << "The improper trigger name is: " << pathSpecifier << "\n";  
      if (noex_demanded &&  exception_spec)
          throw edm::Exception(errors::Configuration)
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request a trigger name starting with exception@ "
	       "and also demanding no &exceptions.\n"
            << "The improper trigger name is: " << pathSpecifier << "\n";  


      // instead of "see if the name can be found in the full list of paths"
      // we want to find all paths that match this name.	
      std::vector<Strings::const_iterator> matches =
	      regexMatch(triggernames, realname);

      if (matches.empty() && !is_glob(realname)) 
      {
          throw edm::Exception(errors::Configuration)
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request a trigger name that does not exist\n"
            << "The unknown trigger name is: " << realname << "\n";  
      }
      if (matches.empty() && is_glob(realname)) 
      {
          LogWarning("Configuration")
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request a wildcarded trigger name that does not match any trigger \n"
            << "The wildcarded trigger name is: " << realname << "\n";  
      }

      if (!negative_criterion && !noex_demanded && !exception_spec) {
	for (unsigned int t = 0; t != matches.size(); ++t) {
	  BitInfo bi(distance(triggernames.begin(),matches[t]), true);
	  absolute_acceptors_.push_back(bi);
	}
      } else if (!negative_criterion && noex_demanded) {
	for (unsigned int t = 0; t != matches.size(); ++t) {
	  BitInfo bi(distance(triggernames.begin(),matches[t]), true);
	  conditional_acceptors_.push_back(bi);
	}
      } else if (exception_spec) {
	for (unsigned int t = 0; t != matches.size(); ++t) {
	  BitInfo bi(distance(triggernames.begin(),matches[t]), true);
	  exception_acceptors_.push_back(bi);
	}
      } else if (negative_criterion && !noex_demanded) {
	if (matches.empty()) {
            throw edm::Exception(errors::Configuration)
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request all fails on a set of trigger names that do not exist\n"
            << "The problematic name is: " << pathSpecifier << "\n";  

	} else if (matches.size() == 1) {
	  BitInfo bi(distance(triggernames.begin(),matches[0]), false);
	  absolute_acceptors_.push_back(bi);
	} else {
	  Bits mustfail;
	  for (unsigned int t = 0; t != matches.size(); ++t) {
	    BitInfo bi(distance(triggernames.begin(),matches[t]), false);
	    // We set this to false because that will demand bits are Fail. 
	    mustfail.push_back(bi);
	  }
	  all_must_fail_.push_back(mustfail);
	} 	
      } else if (negative_criterion && noex_demanded) {
	if (matches.empty()) {
            throw edm::Exception(errors::Configuration)
            << "EventSelector::init, An OutputModule is using SelectEvents\n"
               "to request all fails on a set of trigger names that do not exist\n"
            << "The problematic name is: " << pathSpecifier << "\n";  

	} else if (matches.size() == 1) {
	  BitInfo bi(distance(triggernames.begin(),matches[0]), false);
	  conditional_acceptors_.push_back(bi);
	} else {
	  Bits mustfail;
	  for (unsigned int t = 0; t != matches.size(); ++t) {
	    BitInfo bi(distance(triggernames.begin(),matches[t]), false);
	    mustfail.push_back(bi);
	  }
	  all_must_fail_noex_.push_back(mustfail);
	}
      } 
    } // end of the for loop on i(paths.begin()), end(paths.end())

    if (unrestricted_star && negated_star && exception_star) accept_all_ = true;

    // std::cerr << "### init exited\n";

  } // EventSelector::init
  
  bool EventSelector::acceptEvent(TriggerResults const& tr)
  {
    if (accept_all_) return true;
    
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
      if (!(psetID_initialized_ && psetID_ == tr.parameterSetID())) {

        Strings triggernames;
        bool fromPSetRegistry;

        Service<service::TriggerNamesService> tns;
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
          throw edm::Exception(errors::Unknown)
            << "EventSelector::acceptEvent cannot find the trigger names for\n"
               "a process for which the configuration has requested that the\n"
               "OutputModule use TriggerResults to select events from.  This should\n"
               "be impossible, please send information to reproduce this problem to\n"
               "the edm developers.\n"; 
	}
      }
    }

    // Now make the decision, based on the supplied TriggerResults tr,
    // which of course can be treated as an HLTGlobalStatus by inheritance
    
    return selectionDecision(tr);
    
  } // acceptEvent(TriggerResults const& tr)

  bool 
  EventSelector::acceptEvent(unsigned char const* array_of_trigger_results, 
  			     int number_of_trigger_paths) const
  {

    // This should never occur unless someone uses this function in
    // an incorrect way ...
    if (!results_from_current_process_) {
      throw edm::Exception(errors::Configuration)
        << "\nEventSelector.cc::acceptEvent, you are attempting to\n"
        << "use a bit array for trigger results instead of the\n"
        << "TriggerResults object for a previous process.  This\n"
        << "will not work and ought to be impossible\n";
    }

    if (accept_all_) return true;

    // Form HLTGlobalStatus object to represent the array_of_trigger_results
    HLTGlobalStatus tr(number_of_trigger_paths);
    int byteIndex = 0;
    int subIndex  = 0;
    for (int pathIndex = 0; pathIndex < number_of_trigger_paths; ++pathIndex)
    {
      int state = array_of_trigger_results[byteIndex] >> (subIndex * 2);
      state &= 0x3;
      HLTPathStatus pathStatus(static_cast<hlt::HLTState>(state));
      tr[pathIndex] = pathStatus;
      ++subIndex;
      if (subIndex == 4)
      { ++byteIndex;
        subIndex = 0;
      }
    }    

    // Now make the decision, based on the HLTGlobalStatus tr,
    // which we have created from the supplied array of results
    
    return selectionDecision(tr);

  } // acceptEvent(array_of_trigger_results, number_of_trigger_paths)

  bool 
  EventSelector::selectionDecision(HLTGlobalStatus const& tr) const
  {
    if (accept_all_) return true;

    bool exceptionPresent = false;
    bool exceptionsLookedFor = false;
    
    if (acceptOneBit(absolute_acceptors_, tr)) return true;
    if (acceptOneBit(conditional_acceptors_, tr)) {
      exceptionPresent = containsExceptions(tr);
      if (!exceptionPresent) return true;
      exceptionsLookedFor = true;
    }
    if (acceptOneBit(exception_acceptors_, tr, hlt::Exception)) return true;

    for (std::vector<Bits>::const_iterator f =  all_must_fail_.begin();
    					   f != all_must_fail_.end(); ++f)
    {
      if (acceptAllBits(*f, tr)) return true;
    }
    for (std::vector<Bits>::const_iterator fn =  all_must_fail_noex_.begin();
    					   fn != all_must_fail_noex_.end(); ++fn)
    {
      if (acceptAllBits(*fn, tr)) {
        if (!exceptionsLookedFor) exceptionPresent = containsExceptions(tr);
        return (!exceptionPresent);
      }
    }
    
    // If we have not accepted based on any of the acceptors, nor on any one of
    // the all_must_fail_ collections, then we reject this event.
    
    return false;
  
  }  // selectionDecision()

// Obsolete...
  bool EventSelector::acceptTriggerPath(HLTPathStatus const& pathStatus,
                                        BitInfo const& pathInfo) const
  {
    return (((pathStatus.state()==hlt::Pass) && (pathInfo.accept_state_)) ||
            ((pathStatus.state()==hlt::Fail) && !(pathInfo.accept_state_)) ||
            ((pathStatus.state()==hlt::Exception)));
  }

  // Indicate if any bit in the trigger results matches the desired value
  // at that position, based on the Bits array.  If s is Exception, this
  // looks for a Exceptionmatch; otherwise, true-->Pass, false-->Fail.
  bool 
  EventSelector::acceptOneBit(Bits const& b, 
    		       	       HLTGlobalStatus const& tr, 
    		               hlt::HLTState const& s) const
  {
    bool lookForException = (s == hlt::Exception);
    Bits::const_iterator i(b.begin());
    Bits::const_iterator e(b.end());
    for(;i!=e;++i) {
      hlt::HLTState bstate = 
          lookForException ? hlt::Exception
      			   : i->accept_state_ ? hlt::Pass
				              : hlt::Fail;
      if (tr[i->pos_].state() == bstate) return true;
    }
    return false;    
  } // acceptOneBit			       

  // Indicate if *every* bit in the trigger results matches the desired value
  // at that position, based on the Bits array: true-->Pass, false-->Fail.
  bool 
  EventSelector::acceptAllBits(Bits const& b, 
    		        	HLTGlobalStatus const& tr) const
  {
    Bits::const_iterator i(b.begin());
    Bits::const_iterator e(b.end());
    for(;i!=e;++i) {
      hlt::HLTState bstate = i->accept_state_ ? hlt::Pass : hlt::Fail;
      if (tr[i->pos_].state() != bstate) return false;
    }
    return true;    
  } // acceptAllBits			       

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

	// Finally, check in case the selection element was a wildcarded 
	// negative such as "!*":
	
        if (!oneResultMatched)  {
	  for (unsigned int iPath = 0; iPath < fullTriggerCount; iPath++) {
            sampleResults[iPath] = HLTPathStatus(hlt::Fail, 0);
          }
	  if (evtSelector.acceptEvent(sampleResults)) {
              oneResultMatched = true;
          }
        }
	
        // if none of the possible trigger results matched the
        // selection element, then we declare the whole selection
        // list invalid
        if (!oneResultMatched)
        {
          return false;
        }
      }
      catch (edm::Exception const& excpt)
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
    bool overlap = false;
    
    // first, test that the selection lists are valid
    if (!selectionIsValid(pathspec1, fullTriggerList) ||
        !selectionIsValid(pathspec2, fullTriggerList))
    {
      return evtSel::InvalidSelection;
    }
 
    // catch exceptions from the EventSelector constructor
    // (and anywhere else) and mark those as failures
    try
    {
      // create an EventSelector instance for each selection list
      EventSelector a(pathspec1, fullTriggerList);
      EventSelector b(pathspec2, fullTriggerList);

      unsigned int N = fullTriggerList.size();

      // create the expanded masks for the various decision lists in a and b
      std::vector<bool> 
      	aPassAbs = expandDecisionList(a.absolute_acceptors_,true,N);
      std::vector<bool> 
      	aPassCon = expandDecisionList(a.conditional_acceptors_,true,N);
      std::vector<bool> 
      	aFailAbs = expandDecisionList(a.absolute_acceptors_,false,N);
      std::vector<bool> 
      	aFailCon = expandDecisionList(a.conditional_acceptors_,false,N);
      std::vector<bool> 
      	aExc = expandDecisionList(a.exception_acceptors_,true,N);
      std::vector< std::vector<bool> > aMustFail;
      for (unsigned int m = 0; m != a.all_must_fail_.size(); ++m) {
        aMustFail.push_back(expandDecisionList(a.all_must_fail_[m],false,N));
      }
      std::vector< std::vector<bool> > aMustFailNoex;
      for (unsigned int m = 0; m != a.all_must_fail_noex_.size(); ++m) {
        aMustFailNoex.push_back 
		(expandDecisionList(a.all_must_fail_noex_[m],false,N));
      }

      std::vector<bool> 
      	bPassAbs = expandDecisionList(b.absolute_acceptors_,true,N);
      std::vector<bool> 
      	bPassCon = expandDecisionList(b.conditional_acceptors_,true,N);
      std::vector<bool> 
      	bFailAbs = expandDecisionList(b.absolute_acceptors_,false,N);
      std::vector<bool> 
      	bFailCon = expandDecisionList(b.conditional_acceptors_,false,N);
      std::vector<bool> 
      	bExc = expandDecisionList(b.exception_acceptors_,true,N);
      std::vector< std::vector<bool> > bMustFail;
      for (unsigned int m = 0; m != b.all_must_fail_.size(); ++m) {
        bMustFail.push_back(expandDecisionList(b.all_must_fail_[m],false,N));
      }
      std::vector< std::vector<bool> > bMustFailNoex;
      for (unsigned int m = 0; m != b.all_must_fail_noex_.size(); ++m) {
        bMustFailNoex.push_back 
		(expandDecisionList(b.all_must_fail_noex_[m],false,N));
      }

      std::vector<bool> aPass = combine(aPassAbs, aPassCon);
      std::vector<bool> bPass = combine(bPassAbs, bPassCon);
      std::vector<bool> aFail = combine(aFailAbs, aFailCon);
      std::vector<bool> bFail = combine(bFailAbs, bFailCon);

      // Check for overlap in the primary masks
      overlap = overlapping(aPass, bPass) || 
      		overlapping(aFail, bFail) || 
     		overlapping(aExc, bExc);
      if (overlap) return identical(a,b,N) ? evtSel::ExactMatch 
      					     : evtSel::PartialOverlap;

      // Check for overlap of a primary fail mask with a must fail mask
      for (unsigned int f = 0; f != aMustFail.size(); ++f) {
        overlap = overlapping(aMustFail[f], bFail);
	if (overlap) return evtSel::PartialOverlap;
	for (unsigned int g = 0; g != bMustFail.size(); ++g) {
          overlap = subset(aMustFail[f], bMustFail[g]);
	  if (overlap) return evtSel::PartialOverlap;
	} 
	for (unsigned int g = 0; g != bMustFailNoex.size(); ++g) {
          overlap = subset(aMustFail[f], bMustFailNoex[g]);
	  if (overlap) return evtSel::PartialOverlap;
	}
      }
      for (unsigned int f = 0; f != aMustFailNoex.size(); ++f) {
        overlap = overlapping(aMustFailNoex[f], bFail);
	if (overlap) return evtSel::PartialOverlap;
	for (unsigned int g = 0; g != bMustFail.size(); ++g) {
          overlap = subset(aMustFailNoex[f], bMustFail[g]);
	  if (overlap) return evtSel::PartialOverlap;
	} 
	for (unsigned int g = 0; g != bMustFailNoex.size(); ++g) {
          overlap = subset(aMustFailNoex[f], bMustFailNoex[g]);
	  if (overlap) return evtSel::PartialOverlap;
	}
      }
      for (unsigned int g = 0; g != bMustFail.size(); ++g) {
        overlap = overlapping(bMustFail[g], aFail);
	if (overlap) return evtSel::PartialOverlap;
      }
      for (unsigned int g = 0; g != bMustFailNoex.size(); ++g) {
        overlap = overlapping(bMustFail[g], aFail);
	if (overlap) return evtSel::PartialOverlap;
      }

    }
    catch (edm::Exception const& excpt)
    {
      return evtSel::InvalidSelection;
    }

    // If we get to here without overlap becoming true, there is no overlap

    return evtSel::NoOverlap;

  } // testSelectionOverlap

#ifdef REMOVE
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
    if (!selectionIsValid(pathspec1, fullTriggerList) ||
        !selectionIsValid(pathspec2, fullTriggerList))
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
    catch (edm::Exception const& excpt)
    {
      return evtSel::InvalidSelection;
    }

    if (exactMatch) {return evtSel::ExactMatch;}
    if (noOverlap) {return evtSel::NoOverlap;}
    return evtSel::PartialOverlap;
  }
#endif

  /**
   * Applies a trigger selection mask to a specified trigger result object.
   * Within the trigger result object, each path status is left unchanged
   * if it satisfies the trigger selection (path specs) or cleared if it
   * does not satisfy the trigger selection.  In this way, the resulting
   * trigger result object contains only path status values that "pass"
   * the selection criteria.
   *
   * @param inputResults The raw trigger results object that will be masked.
   * @return a copy of the input trigger results object with only the path
   *         status results that match the trigger selection.
   * @throws edm::Exception if the number of paths in the TriggerResults
   *         object does not match the specified full trigger list, or
   *         if the trigger selection is invalid in the context of the
   *         full trigger list.
   */
  boost::shared_ptr<TriggerResults>
  EventSelector::maskTriggerResults(TriggerResults const& inputResults)
  {
    // fetch and validate the total number of paths
    unsigned int fullTriggerCount = nTriggerNames_;
    unsigned int N = fullTriggerCount;
    if (fullTriggerCount != inputResults.size())
    {
      throw edm::Exception(errors::EventCorruption)
        << "EventSelector::maskTriggerResults, the TriggerResults\n"
        << "size (" << inputResults.size()
        << ") does not match the number of paths in the\n"
        << "full trigger list (" << fullTriggerCount << ").\n";
    }

    // create a suitable global status object to work with, all in Ready state
    HLTGlobalStatus mask(fullTriggerCount);
    
    // Deal with must_fail acceptors that would cause selection
    for (unsigned int m = 0; m < this->all_must_fail_.size(); ++m) {
      std::vector<bool>  
        f = expandDecisionList(this->all_must_fail_[m],false,N);
      bool all_fail = true;
      for (unsigned int ipath = 0; ipath < N; ++ipath) {        
	if  ((f[ipath]) && (inputResults [ipath].state() != hlt::Fail)) { 
	  all_fail = false;
	  break;
	}
      }
      if (all_fail) {
	for (unsigned int ipath = 0; ipath < N; ++ipath) {
          if  (f[ipath]) { 
	    mask[ipath] = hlt::Fail;
	  }
	}
      }
    }
    for (unsigned int m = 0; m < this->all_must_fail_noex_.size(); ++m) {
      std::vector<bool>  
        f = expandDecisionList(this->all_must_fail_noex_[m],false,N);
      bool all_fail = true;
      for (unsigned int ipath = 0; ipath < N; ++ipath) {        
	if ((f[ipath]) && (inputResults [ipath].state() != hlt::Fail)) { 
	  all_fail = false;
	  break;
	}
      }
      if (all_fail) {
	for (unsigned int ipath = 0; ipath < N; ++ipath) {
          if  (f[ipath]) { 
	    mask[ipath] = hlt::Fail;
	  }
	}
      }
    } // factoring opportunity - work done for fail_noex_ is same as for fail_
    
    // Deal with normal acceptors that would cause selection
    std::vector<bool> 
      aPassAbs = expandDecisionList(this->absolute_acceptors_,true,N);
    std::vector<bool> 
      aPassCon = expandDecisionList(this->conditional_acceptors_,true,N);
    std::vector<bool> 
      aFailAbs = expandDecisionList(this->absolute_acceptors_,false,N);
    std::vector<bool> 
      aFailCon = expandDecisionList(this->conditional_acceptors_,false,N);
    std::vector<bool> 
      aExc = expandDecisionList(this->exception_acceptors_,true,N);
    for (unsigned int ipath = 0; ipath < N; ++ipath) {
      hlt::HLTState s = inputResults [ipath].state();  
      if (((aPassAbs[ipath]) && (s == hlt::Pass))
      		||
	  ((aPassCon[ipath]) && (s == hlt::Pass))		
      		||
	  ((aFailAbs[ipath]) && (s == hlt::Fail))		
      		||
	  ((aFailCon[ipath]) && (s == hlt::Fail))
	        ||
	  ((aExc[ipath]) && (s == hlt::Exception)))
      {
        mask[ipath] = s;
      }		
    }
 
    // Based on the global status for the mask, create and return a 
    // TriggerResults
    boost::shared_ptr<TriggerResults>
      maskedResults(new TriggerResults(mask, inputResults.parameterSetID()));
    return maskedResults;
  }  // maskTriggerResults





#ifdef REMOVE
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
      throw edm::Exception(errors::EventCorruption)
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
      if (!selector.wantAll() && !selector.acceptEvent(sampleResults))
      {
        maskedResults->reset(iPath);
      }
      sampleResults.reset(iPath);
    }
    return maskedResults;
  }
#endif

  /**
   * Returns the list of strings that correspond to the trigger
   * selection request in the specified parameter set (the list
   * of strings contained in the "SelectEvents" parameter).
   *
   * @param pset The ParameterSet that contains the trigger selection.
   * @return the trigger selection list (vector of string).
   */
  std::vector<std::string>
  EventSelector::getEventSelectionVString(ParameterSet const& pset)
  {
    // default the selection to everything (wildcard)
    Strings selection;
    selection.push_back("*");
    selection.push_back("!*");
    selection.push_back("exception@*");

    // the SelectEvents parameter is a ParameterSet within
    // a ParameterSet, so we have to pull it out twice
    ParameterSet selectEventsParamSet =
      pset.getUntrackedParameter("SelectEvents", ParameterSet());
    if (!selectEventsParamSet.empty()) {
      Strings path_specs = 
        selectEventsParamSet.getParameter<Strings>("SelectEvents");
      if (!path_specs.empty()) {
        selection = path_specs;
      }
    }

    // return the result
    return selection;
  }

  bool EventSelector::containsExceptions(HLTGlobalStatus const& tr) const
  {
    unsigned int e = tr.size();
    for (unsigned int i = 0; i < e; ++i) {
      if (tr[i].state() == hlt::Exception) return true;
    }
    return false;
  }

  // The following routines are helpers for testSelectionOverlap
  
  bool 
  EventSelector::identical(std::vector<bool> const& a, 
  			   std::vector<bool> const& b) {
     unsigned int n = a.size();
     if (n != b.size()) return false;
     for (unsigned int i=0; i!=n; ++i) {
       if (a[i] != b[i]) return false;
     }
     return true;
  }
  
  bool 
  EventSelector::identical(EventSelector const& a, 
  			   EventSelector const& b,
			   unsigned int N) 
  {
        // create the expanded masks for the various decision lists in a and b
    if (!identical(expandDecisionList(a.absolute_acceptors_,true,N),
                   expandDecisionList(b.absolute_acceptors_,true,N))) 
		   return false;
    if (!identical(expandDecisionList(a.conditional_acceptors_,true,N),
                   expandDecisionList(b.conditional_acceptors_,true,N))) 
		   return false;
    if (!identical(expandDecisionList(a.absolute_acceptors_,false,N),
                   expandDecisionList(b.absolute_acceptors_,false,N)))
		   return false;
    if (!identical(expandDecisionList(a.conditional_acceptors_,false,N),
                   expandDecisionList(b.conditional_acceptors_,false,N))) 
		   return false;
    if (!identical(expandDecisionList(a.exception_acceptors_,true,N),
                   expandDecisionList(b.exception_acceptors_,true,N)))
		   return false;
    if (a.all_must_fail_.size() != b.all_must_fail_.size()) return false;
    
    std::vector< std::vector<bool> > aMustFail;
    for (unsigned int m = 0; m != a.all_must_fail_.size(); ++m) {
      aMustFail.push_back(expandDecisionList(a.all_must_fail_[m],false,N));
    }
    std::vector< std::vector<bool> > aMustFailNoex;
    for (unsigned int m = 0; m != a.all_must_fail_noex_.size(); ++m) {
      aMustFailNoex.push_back 
	      (expandDecisionList(a.all_must_fail_noex_[m],false,N));
    }
    std::vector< std::vector<bool> > bMustFail;
    for (unsigned int m = 0; m != b.all_must_fail_.size(); ++m) {
      bMustFail.push_back(expandDecisionList(b.all_must_fail_[m],false,N));
    }
    std::vector< std::vector<bool> > bMustFailNoex;
    for (unsigned int m = 0; m != b.all_must_fail_noex_.size(); ++m) {
      bMustFailNoex.push_back 
	      (expandDecisionList(b.all_must_fail_noex_[m],false,N));
    }
    
    for (unsigned int m = 0; m != aMustFail.size(); ++m) {
      bool match = false;
      for (unsigned int k = 0; k != bMustFail.size(); ++k) {
        if (identical(aMustFail[m],bMustFail[k])) {
          match = true;
	  break;
	}
      }
      if (!match) return false;
    }
    for (unsigned int m = 0; m != aMustFailNoex.size(); ++m) {
      bool match = false;
      for (unsigned int k = 0; k != bMustFailNoex.size(); ++k) {
         if (identical(aMustFailNoex[m],bMustFailNoex[k])) {
          match = true;
	  break;
	}
      }
      if (!match) return false;
    }

    return true;
    
  } // identical (EventSelector, EventSelector, N);
  
  std::vector<bool> 
  EventSelector::expandDecisionList(Bits const& b,  
				      bool PassOrFail,
				      unsigned int n)
  {
    std::vector<bool> x(n, false);
    for (unsigned int i = 0; i != b.size(); ++i) {
      if (b[i].accept_state_ == PassOrFail) x[b[i].pos_] = true;
    }
    return x;
  } // expandDecisionList	
  
  // Determines whether a and b share a true bit at any position
  bool EventSelector::overlapping(std::vector<bool> const& a, 
    			             std::vector<bool> const& b)
  {
    if (a.size() != b.size()) return false;
    for (unsigned int i = 0; i != a.size(); ++i) {
      if (a[i] && b[i]) return true;
    }
    return false;
  } // overlapping
  
  // determines whether the true bits of a are a non-empty subset of those of b,
  // or vice-versa.  The subset need not be proper.
  bool EventSelector::subset(std::vector<bool> const& a, 
    			       std::vector<bool> const& b)
  {
    if (a.size() != b.size()) return false;
    // First test whether a is a non-empty subset of b 
    bool aPresent = false;
    bool aSubset = true;
    for (unsigned int i = 0; i != a.size(); ++i) {
      if (a[i]) {
        aPresent = true;
        if (!b[i]) {
	  aSubset = false;
	  break; 
	}
      }
    }   
    if (!aPresent) return false;
    if (aSubset) return true;
    
    // Now test whether b is a non-empty subset of a 
    bool bPresent = false;
    bool bSubset = true;
    for (unsigned int i = 0; i != b.size(); ++i) {
      if (b[i]) {
        bPresent = true;
        if (!a[i]) {
	  bSubset = false;
	  break; 
	}
      }
    }   
    if (!bPresent) return false;
    if (bSubset) return true;
 
    return false;     				     
  } // subset
  
  // Creates a vector of bits which is the OR of a and b
  std::vector<bool> 
  EventSelector::combine(std::vector<bool> const& a, 
    			  std::vector<bool> const& b)
  {
    assert(a.size() == b.size());
    std::vector<bool> x(a.size());
    for (unsigned int i = 0; i != a.size(); ++i) {
      x[i] = a[i] || b[i];
    } // a really sharp compiler will optimize the hell out of this, 
      // exploiting word-size OR operations.
    return x;
  } // combine			   			      

}
