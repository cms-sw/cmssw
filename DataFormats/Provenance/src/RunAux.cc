#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"

/*----------------------------------------------------------------------

$Id: RunAux.cc,v 1.1 2007/03/04 04:48:10 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(RunAux const& from, RunAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = RunID(from.id_);
  }
}
