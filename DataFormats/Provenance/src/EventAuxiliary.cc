#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: EventAuxiliary.cc,v 1.3 2007/07/18 20:12:04 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
    os << "LuminosityBlockNumber_t = " << luminosityBlock_ << std::endl;
  }

  bool
  isSameEvent(EventAuxiliary const& a, EventAuxiliary const& b) {
    return
      a.id_ == b.id_ &&
      a.processGUID_ == b.processGUID_ &&
      a.luminosityBlock_ == b.luminosityBlock_ &&
      a.time_ == b.time_ &&
      a.isRealData_ == b.isRealData_ &&
      a.experimentType_ == b.experimentType_ &&
      a.bunchCrossing_ == b.bunchCrossing_ &&
      a.storeNumber_ == b.storeNumber_;
  }
}
