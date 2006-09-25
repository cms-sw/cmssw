
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/bind.hpp"

#include <algorithm>

typedef std::vector<std::string> Strings;
using namespace std;


namespace edm
{

  EventSelector::EventSelector(edm::ParameterSet const& pset,
			       std::string const& process_name,
			       Strings const& names):
    process_name_(process_name),
    accept_all_(false),
    decision_bits_()
  {
    edm::ParameterSet def;
    edm::ParameterSet mine =
      pset.getUntrackedParameter<ParameterSet>("SelectEvents",def);

    if(mine==def)
      {
	// no criteria present, accept all
	accept_all_ = true;
	return;
      }

    Strings paths = mine.getParameter<Strings>("SelectEvents");

    if(paths.empty())
      {
	accept_all_ = true;
	return;
      }

    Strings::iterator i(paths.begin()),end(paths.end());
    bool star_done = false, not_star_done = false;
    for(;i!=end;++i)
      {
	// remove any whitespace
	string::iterator newend = remove(i->begin(),i->end(),' ');
	string tmp(i->begin(),newend);

	if(tmp == "*" && !star_done)
	  {
	    star_done = true;
	    for(unsigned int k=0;k<names.size();++k) 
	      decision_bits_.push_back(BitInfo(k,true));
	  }
	else if(tmp == "!*" && !not_star_done)
	  {
	    not_star_done = true;
	    for(unsigned int k=0;k<names.size();++k) 
	      decision_bits_.push_back(BitInfo(k,false));
	  }
	else
	  {
	    // brute force algorthn here, assumes arrays are small
	    // and only passed through during initialization...

	    bool accept_level = tmp[0]=='!' ? false : true;
	    // make the name without the bang if need be
	    string const& realname = 
	      accept_level ? tmp : string((tmp.begin()+1),tmp.end());

	    // see if the name can be found in the full list of paths
	    Strings::const_iterator pos = 
	      find(names.begin(),names.end(),realname);
	    if(pos!=names.end())
	      {
		BitInfo bi(distance(names.begin(),pos),accept_level);
		decision_bits_.push_back(bi);
	      }
	    else
	      {
		LogWarning("configuration")
		  << "EventSelector: a trigger path named " << tmp
		  << "is not available in the current process.\n";
	      }
	  }
      }

	  if(not_star_done && star_done)
	  	accept_all_ = true;
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
