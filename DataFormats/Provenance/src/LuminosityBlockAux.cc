#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"

/*----------------------------------------------------------------------

$Id: LuminosityBlockAux.cc,v 1.1 2007/03/15 21:45:37 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(LuminosityBlockAux const& from, LuminosityBlockAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = LuminosityBlockID(from.runID_, from.id_);
  }
}
