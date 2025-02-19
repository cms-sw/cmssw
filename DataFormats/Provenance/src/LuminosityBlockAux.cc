#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

/*----------------------------------------------------------------------

$Id: LuminosityBlockAux.cc,v 1.3 2007/07/31 23:58:53 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(LuminosityBlockAux const& from, LuminosityBlockAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = LuminosityBlockID(from.runID_, from.id_);
    to.beginTime_ = to.endTime_ = Timestamp::invalidTimestamp();
  }
}
