#ifndef Common_TriggerResults_h
#define Common_TriggerResults_h

/** \class TriggerResults
 *
 *  Original Author: Jim Kowalkowski 13-01-06
 *  $Id: TriggerResults.h,v 1.1 2006/02/08 00:44:23 wmtan Exp $
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
 *  $Date: 2006/04/11 10:10:10 $
 *  $Revision: 1.0 $
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
    edm::ParameterSetID id_;
    Strings             triggernames_;

  public:
    TriggerResults() : HLTGlobalStatus(), id_(), triggernames_() {}

    TriggerResults(const HLTGlobalStatus& hlt, const Strings& triggernames)
      : HLTGlobalStatus(hlt), id_(), triggernames_(triggernames) { }

    TriggerResults(const HLTGlobalStatus& hlt, const edm::ParameterSetID& id,  const Strings& triggernames)
      : HLTGlobalStatus(hlt), id_(id), triggernames_(triggernames) {
    assert (hlt.size()==triggernames.size());
    }

    //

    const std::string& name(unsigned int i) const {return triggernames_.at(i);}

    unsigned int find (const std::string name) const {
      const unsigned int n(size());
      for (unsigned int i=0; i!=n; i++) if (triggernames_[i]==name) return i;
      return n;
    }

  };

}

#endif
