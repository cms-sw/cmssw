#ifndef Common_TriggerResults_h
#define Common_TriggerResults_h

/** \class TriggerResults
 *
 *  Original Author: Jim Kowalkowski 13-01-06
 *  $Id: TriggerResults.h,v 1.4 2007/01/05 18:41:04 wdd Exp $
 *
 *  The trigger path results are maintained here as a sequence of
 *  entries, one per trigger path.  They are assigned in the order
 *  they appeared in the process-level pset.
 *
 *  Implementation note: there is as of this writing, no place in the
 *  file to store parameter sets or run information.  The trigger bit
 *  descriptions need to be stored in these sections.  This object
 *  stores the parameter ID, which can be used to locate the parameter
 *  in the file when that option becomes available.  For now, this
 *  object contains the trigger path names as a vector of strings.
 *
 *  $Date: 2007/01/05 18:41:04 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/Common/interface/ParameterSetID.h"

#include <string>
#include <vector>
#include<cassert>

namespace edm
{
  class TriggerResults : public HLTGlobalStatus {

    typedef std::vector<std::string> Strings;

  private:
    edm::ParameterSetID psetid_;
    Strings             names_;

  public:
    TriggerResults() : HLTGlobalStatus(), psetid_(), names_() {}

    TriggerResults(const HLTGlobalStatus& hlt, const Strings& names)
      : HLTGlobalStatus(hlt), psetid_(), names_(names) { }

    TriggerResults(const HLTGlobalStatus& hlt, const edm::ParameterSetID& psetid,  const Strings& names)
      : HLTGlobalStatus(hlt), psetid_(psetid), names_(names) {
    assert (hlt.size()==names.size());
    }

    const std::vector<std::string>& getTriggerNames() const { return names_; }

    const std::string& name(unsigned int i) const {return names_.at(i);}

    unsigned int find (const std::string& name) const {
      const unsigned int n(size());
      for (unsigned int i = 0; i != n; ++i) if (names_[i] == name) return i;
      return n;
    }

  };

}

#endif
