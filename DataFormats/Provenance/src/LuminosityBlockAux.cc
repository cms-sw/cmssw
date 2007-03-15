#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

/*----------------------------------------------------------------------

$Id: LuminosityBlockAux.cc,v 1.1 2007/03/04 04:48:10 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(LuminosityBlockAux const& from, LuminosityBlockAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = LuminosityBlockID(from.runID_, from.id_);
  }
}
