#ifndef DataFormats_Common_TriggerResults_h
#define DataFormats_Common_TriggerResults_h

/** \class edm::TriggerResults
 *
 *  Original Authors: Jim Kowalkowski 13-01-06
 *                    Martin Grunewald
 *  $Id: TriggerResults.h,v 1.10 2008/05/16 00:29:10 paterno Exp $
 *
 *  The trigger path results are maintained here as a sequence of
 *  entries, one per trigger path.  They are assigned in the order
 *  they appeared in the process-level pset.  (They are actually
 *  stored in the base class HLTGlobalStatus)
 *
 *  The ParameterSetID can be used to get a ParameterSet from
 *  the registry of parameter sets.  This ParameterSet contains
 *  a vector<string> named "trigger_paths" that contains the
 *  trigger path names in the same order as the trigger path
 *  results stored here.
 *
 *  The vector<string> contained in this class is empty and
 *  no longer used.  It is kept for backward compatibility reasons.
 *  In early versions of the code, the trigger results paths names
 *  were stored there.
 *  NOTE: In the latest code, this data member has been removed.
 *
 */

#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"


#include <string>
#include <vector>

namespace edm
{
  class TriggerResults : public HLTGlobalStatus, public DoNotRecordParents  {

    typedef std::vector<std::string> Strings;

  private:
    /// Parameter set id
    edm::ParameterSetID psetid_;


  public:

    /// Trivial contructor
    TriggerResults() : HLTGlobalStatus(), psetid_() { }

    /// Standard contructor
    TriggerResults(const HLTGlobalStatus& hlt, const edm::ParameterSetID& psetid)
      : HLTGlobalStatus(hlt), psetid_(psetid) { }


    /// Get stored parameter set id
    const ParameterSetID& parameterSetID() const { return psetid_; }

    /// swap function
    void swap(TriggerResults& other) {
      this->HLTGlobalStatus::swap(other);
      psetid_.swap(other.psetid_);
    }

    /// Copy assignment using swap.
    TriggerResults& operator=(TriggerResults const& rhs) {
      TriggerResults temp(rhs);
      this->swap(temp);
      return *this;
    }

  };

  // Free swap function
  inline
  void
  swap(TriggerResults& lhs, TriggerResults& rhs) {
    lhs.swap(rhs);
  }
}

#endif
