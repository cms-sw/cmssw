#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

/*----------------------------------------------------------------------


----------------------------------------------------------------------*/

namespace edm {
  void conversion(RunAux const& from, RunAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = RunID(from.id_);
    to.beginTime_ = to.endTime_ = Timestamp::invalidTimestamp(); 
  }
}
