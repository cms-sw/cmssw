#ifndef Framework_EventSelector_h
#define Framework_EventSelector_h

/*
  Author: Jim Kowalkowski 01-02-06
  $Id: EventSelector.h,v 1.8 2007/06/15 18:41:46 wdd Exp $

 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <vector>
#include <string>

namespace edm
{
  class EventSelector
  {
  public:

    typedef std::vector<std::string> Strings;

    EventSelector(Strings const& pathspecs,
		  Strings const& names);

    explicit
    EventSelector(Strings const& pathspecs);

    EventSelector(edm::ParameterSet const& pset,
		  Strings const& triggernames);

    bool wantAll() const { return accept_all_; }
    bool acceptEvent(TriggerResults const&);
    bool acceptEvent(unsigned char const*, int) const;

  private:

    void init(Strings const& paths,
	      Strings const& triggernames);

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

    bool psetID_initialized_;
    ParameterSetID psetID_;

    Strings paths_;

    int nTriggerNames_;
    bool notStarPresent_;

    bool acceptTriggerPath(HLTPathStatus const&, BitInfo const&) const;
  };
}

#endif
