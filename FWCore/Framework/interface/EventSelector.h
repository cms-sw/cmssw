#ifndef Framework_EventSelector_h
#define Framework_EventSelector_h

/*
  Author: Jim Kowalkowski 01-02-06
  $Id: EventSelector.h,v 1.5 2006/09/25 19:55:18 paterno Exp $

 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include <vector>
#include <string>

namespace edm
{
  class EventSelector
  {
  public:
    EventSelector(std::vector<std::string> const& pathspecs,
		  std::vector<std::string> const& names);

    EventSelector(edm::ParameterSet const& pset,
		  std::string const& processname,
		  std::vector<std::string> const& triggernames);

    EventSelector(edm::ParameterSet const& pset,
		  std::vector<std::string> const& triggernames);

    std::string getProcessName() const { return process_name_; }
    bool wantAll() const { return accept_all_; }
    bool acceptEvent(TriggerResults const&) const;
    bool acceptEvent(unsigned char const*, int) const;

  private:

    void init(std::vector<std::string> const& paths,
	      std::vector<std::string> const& triggernames);

    struct BitInfo
    {
      BitInfo(unsigned int pos, bool state):pos_(pos),accept_state_(state) { }
      BitInfo():pos_(),accept_state_() { }

      unsigned int pos_;
      bool accept_state_;
    };

    typedef std::vector<BitInfo> Bits;

    std::string process_name_;
    bool accept_all_;
    Bits decision_bits_;

    bool acceptTriggerPath(HLTPathStatus const&, BitInfo const&) const;
  };
}


#endif

