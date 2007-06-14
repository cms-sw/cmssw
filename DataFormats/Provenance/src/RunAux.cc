#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"

/*----------------------------------------------------------------------

$Id: RunAux.cc,v 1.1 2007/03/15 21:45:37 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(RunAux const& from, RunAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = RunID(from.id_);
  }
}
