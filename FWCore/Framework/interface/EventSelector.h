#ifndef Framework_EventSelector_h
#define Framework_EventSelector_h

/*
  Author: Jim Kowalkowski 01-02-06
  $Id: EventSelector.h,v 1.4 2006/09/07 13:49:26 biery Exp $

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
    EventSelector(edm::ParameterSet const& pset,
		  std::string const& process_name,
		  std::vector<std::string> const& names);

    std::string getProcessName() const { return process_name_; }
    bool wantAll() const { return accept_all_; }
    bool acceptEvent(TriggerResults const&) const;
    bool acceptEvent(unsigned char const*, int) const;

  private:

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

