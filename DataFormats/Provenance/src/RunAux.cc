#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

/*----------------------------------------------------------------------

$Id: RunAux.cc,v 1.3 2007/07/31 23:58:53 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(RunAux const& from, RunAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = RunID(from.id_);
    to.beginTime_ = to.endTime_ = Timestamp::invalidTimestamp(); 
  }
}
