#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include <cassert>
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  void
  LuminosityBlockAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
  }

  void
  LuminosityBlockAuxiliary::mergeAuxiliary(LuminosityBlockAuxiliary const& newAux) {
    assert(id_ == newAux.id_);
    assert(processHistoryID_ == newAux.processHistoryID_);
    if (beginTime_ == Timestamp::invalidTimestamp() ||
        newAux.beginTime() == Timestamp::invalidTimestamp()) {
      beginTime_ = Timestamp::invalidTimestamp();
    }
    else if (newAux.beginTime() < beginTime_) {
      beginTime_ = newAux.beginTime();
    }
   
    if (endTime_ == Timestamp::invalidTimestamp() ||
        newAux.endTime() == Timestamp::invalidTimestamp()) {
      endTime_ = Timestamp::invalidTimestamp();
    }
    else if (newAux.endTime() > endTime_) {
      endTime_ = newAux.endTime();
    }

  }
}
