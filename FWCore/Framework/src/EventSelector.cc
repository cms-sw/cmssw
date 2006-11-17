
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/bind.hpp"
#include "boost/algorithm/string.hpp"

#include <algorithm>

typedef std::vector<std::string> stringvec;
using namespace std;


namespace edm
{
  EventSelector::EventSelector(stringvec const& pathspecs,
			       stringvec const& names):
    process_name_(),
    accept_all_(false),
    decision_bits_()
  {
    init(pathspecs, names);
  }

  void
  EventSelector::init(stringvec const& paths,
		      stringvec const& triggernames)
  {
    if ( paths.empty() )
      {
	accept_all_ = true;
	return;
      }

    bool star_done = false;
    bool not_star_done = false;
    for (stringvec::const_iterator i(paths.begin()), end(paths.end()); 
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
	else if(current_path == "!*" && !not_star_done)
	  {
	    not_star_done = true;
	    for(unsigned int k=0;k<triggernames.size();++k) 
	      decision_bits_.push_back(BitInfo(k,false));
	  }
	else
	  {
	    // brute force algorthim here, assumes arrays are small
	    // and only passed through during initialization...

	    bool accept_level = (current_path[0]!='!');
	    // make the name without the bang if need be
	    string const& realname = 
	      accept_level 
	      ? current_path 
	      : string((current_path.begin()+1), current_path.end());
	    
	    // see if the name can be found in the full list of paths
	    stringvec::const_iterator pos = 
	      find(triggernames.begin(),triggernames.end(),realname);
	    if(pos!=triggernames.end())
	      {
		BitInfo bi(distance(triggernames.begin(),pos),accept_level);
		decision_bits_.push_back(bi);
	      }
	    else
	      {
		LogWarning("configuration")
		  << "EventSelector: a trigger path named " << current_path
		  << "is not available in the current process.\n";
	      }
	  }
      }
    
    if (not_star_done && star_done) accept_all_ = true;
  }
  
  EventSelector::EventSelector(ParameterSet const& config,
			       string const& processname,
			       stringvec const& triggernames) :
    process_name_(processname),
    accept_all_(false),
    decision_bits_()
  {
    stringvec paths; // default is empty...

    if (!config.empty())
      paths = config.getParameter<stringvec>("SelectEvents");

    init(paths, triggernames);
  }
  
  // If there were a way to just call another c'tor, we'd call:
  //
  //       EventSelector(config, "PROD", names) 
  //
  // but C++ does not (yet) support this, so we have to do it
  // ourselves...

  EventSelector::EventSelector(edm::ParameterSet const& config,
			       stringvec const& triggernames):
    process_name_("PROD"),
    accept_all_(false),
    decision_bits_()
  {
    stringvec paths; // default is empty...

    if (!config.empty())
      paths = config.getParameter<stringvec>("SelectEvents");

    init(paths, triggernames);
  }


  bool EventSelector::acceptEvent(TriggerResults const& tr) const
  {
    Bits::const_iterator i(decision_bits_.begin()),e(decision_bits_.end());
    for(;i!=e;++i)
      {
        if ( this->acceptTriggerPath(tr[i->pos_], *i) )
          {
            return true;
          }
      }
    return false;
  }

  bool EventSelector::acceptEvent(unsigned char const* array_of_trigger_results, int number_of_trigger_paths) const
  {
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
