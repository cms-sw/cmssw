#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
  }

  bool
  isSameEvent(EventAuxiliary const& a, EventAuxiliary const& b) {
    return
      a.id_ == b.id_ &&
      a.processGUID_ == b.processGUID_ &&
      a.luminosityBlock() == b.luminosityBlock() &&
      a.time_ == b.time_ &&
      a.isRealData_ == b.isRealData_ &&
      a.experimentType_ == b.experimentType_ &&
      a.bunchCrossing_ == b.bunchCrossing_ &&
      a.storeNumber_ == b.storeNumber_;
  }
}
