#ifndef Framework_EventSelector_h
#define Framework_EventSelector_h

/*
  Author: Jim Kowalkowski 01-02-06

 */

// Change Log
//
// 1 - Mark Fischler Feb 6, 2008
//	Internals for implementation of glob-style wildcard selection 
//	In particular, !xyz* requires the vector nonveto_bits_
//	nonveto_bits_ is designed to also accomodate an AND of triggers
//      selection criterion, if that is wanted at some future date.

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <string>

namespace edm
{
  // possible return codes for the testSelectionOverlap
  // method defined below.
  namespace evtSel
  {
    enum OverlapResult {InvalidSelection = 0,
                        NoOverlap = 1,
                        PartialOverlap = 2,
                        ExactMatch = 3};
  }

  class ParameterSetDescription;
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

    // 29-Jan-2008, KAB - added methods for testing and using
    // trigger selections (pathspecs).
    static bool selectionIsValid(Strings const& pathspec,
                                 Strings const& fullTriggerList);
    static evtSel::OverlapResult
      testSelectionOverlap(Strings const& pathspec1,
                           Strings const& pathspec2,
                           Strings const& fullTriggerList);
    boost::shared_ptr<TriggerResults>
      maskTriggerResults(TriggerResults const& inputResults);
    static std::vector<std::string>
      getEventSelectionVString(edm::ParameterSet const& pset);

    static void fillDescription(ParameterSetDescription& desc);

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
    Bits absolute_acceptors_;					// change 3
    Bits conditional_acceptors_;				// change 3
    Bits exception_acceptors_;					// change 3
    std::vector<Bits> all_must_fail_;				// change 1
    std::vector<Bits> all_must_fail_noex_;			// change 3

    bool results_from_current_process_;
    bool psetID_initialized_;
    ParameterSetID psetID_;

    Strings paths_;

    int nTriggerNames_;
    bool notStarPresent_;

    bool acceptTriggerPath(HLTPathStatus const&, BitInfo const&) const;

    bool acceptOneBit (Bits const & b, 
    		       HLTGlobalStatus const & tr, 
    		       hlt::HLTState const & s = hlt::Ready) const;
    bool acceptAllBits (Bits const & b, 
    		        HLTGlobalStatus const & tr) const;

    bool containsExceptions(HLTGlobalStatus const & tr) const;
    
    bool selectionDecision(HLTGlobalStatus const & tr) const;
    
    static std::string glob2reg(std::string const& s);
    static std::vector< Strings::const_iterator > 
      matching_triggers(Strings const& trigs, std::string const& s);
      
    static bool identical (std::vector<bool> const & a, 
    			   std::vector<bool> const & b); 
    static bool identical (EventSelector const & a, 
    			   EventSelector const & b,
			   unsigned int N); 
    static std::vector<bool> expandDecisionList ( 
    		Bits const & b,  
		bool PassOrFail,
		unsigned int n);
    static bool overlapping ( std::vector<bool> const& a, 
    			      std::vector<bool> const& b );
    static bool subset  ( std::vector<bool> const& a, 
    			  std::vector<bool> const& b );
    static std::vector<bool> combine ( std::vector<bool> const& a, 
    			               std::vector<bool> const& b );
  };
}

#endif
