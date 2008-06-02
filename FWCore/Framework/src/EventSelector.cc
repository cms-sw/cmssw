
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "boost/algorithm/string.hpp"

#include <algorithm>

using namespace std;


namespace edm
{
  EventSelector::EventSelector(Strings const& pathspecs,
			       Strings const& names):
    accept_all_(false),
    decision_bits_(),
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

	string current_path(*i);
	boost::erase_all(current_path, " \t");
	if(current_path == "*" && !star_done)
	  {
	    star_done = true;
	    for(unsigned int k=0;k<triggernames.size();++k) 
	      decision_bits_.push_back(BitInfo(k,true));
	  }
	else if (current_path == "!*")
	  {
            notStarPresent_ = true;
	  }
	else
	  {
	    // brute force algorithm here, assumes arrays are small
	    // and only passed through during initialization...

	    bool accept_level = (current_path[0]!='!');
	    // make the name without the bang if need be
	    string const& realname = 
	      accept_level 
	      ? current_path 
	      : string((current_path.begin()+1), current_path.end());
	    
	    // see if the name can be found in the full list of paths
	    Strings::const_iterator pos = 
	      find(triggernames.begin(),triggernames.end(),realname);
	    if(pos!=triggernames.end())
	      {
		BitInfo bi(distance(triggernames.begin(),pos),accept_level);
		decision_bits_.push_back(bi);
	      }
	    else
	      {
                throw edm::Exception(edm::errors::Configuration)
                  << "EventSelector::init, An OutputModule is using SelectEvents\n"
                     "to request a trigger name that does not exist\n"
                  << "The unknown trigger name is: " << realname << "\n";  
	      }
	  }
      }
    
    if (notStarPresent_ && star_done) accept_all_ = true;
  }
  
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
    
    // handle the special "!*" case, this selects events that fail all trigger paths
    if (notStarPresent_) {

      bool allFail = true;
      for (int j = 0; j < nTriggerNames_; ++j) {
        if (this->acceptTriggerPath(tr[j], BitInfo(j, true))) allFail = false;
      }
      if (allFail) return true;
    }

    return false;
  }

  bool EventSelector::acceptEvent(unsigned char const* array_of_trigger_results, int number_of_trigger_paths) const
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

    // handle the special "!*" case, this selects events that fail all trigger paths
    if (notStarPresent_) {

      bool allFail = true;
      for (int j = 0; j < nTriggerNames_; ++j) {

        int pathIndex = j;
        if (pathIndex < number_of_trigger_paths)
        {
          int byteIndex = ((int) pathIndex / 4);
          int subIndex = pathIndex % 4;
          int state = array_of_trigger_results[byteIndex] >> (subIndex * 2);
          state &= 0x3;
          HLTPathStatus pathStatus(static_cast<hlt::HLTState>(state));

          if (this->acceptTriggerPath(pathStatus, BitInfo(j, true))) allFail = false;
        }
      }
      if (allFail) return true;
    }

    return false;
  }

  bool EventSelector::acceptTriggerPath(HLTPathStatus const& pathStatus,
                                        BitInfo const& pathInfo) const
  {
    return ( ((pathStatus.state()==hlt::Pass) &&  (pathInfo.accept_state_)) ||
             ((pathStatus.state()==hlt::Fail) && !(pathInfo.accept_state_)) ||
             ((pathStatus.state()==hlt::Exception)) );
  }
}
