#ifndef Framework_EventSelector_h
#define Framework_EventSelector_h

/*
  Author: Jim Kowalkowski 01-02-06
  $Id: EventSelector.h,v 1.6 2006/11/17 23:05:00 paterno Exp $

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

    explicit
    EventSelector(std::vector<std::string> const& pathspecs);

    EventSelector(edm::ParameterSet const& pset,
		  std::vector<std::string> const& triggernames);

    bool wantAll() const { return accept_all_; }
    bool acceptEvent(TriggerResults const&);
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

    bool accept_all_;
    Bits decision_bits_;
    bool results_from_current_process_;

    std::vector<std::string> paths_;

    bool acceptTriggerPath(HLTPathStatus const&, BitInfo const&) const;
  };
}

#endif

