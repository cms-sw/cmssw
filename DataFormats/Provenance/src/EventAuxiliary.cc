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
      a.id() == b.id() &&
      a.processGUID() == b.processGUID() &&
      a.luminosityBlock() == b.luminosityBlock() &&
      a.time() == b.time() &&
      a.isRealData() == b.isRealData() &&
      a.experimentType() == b.experimentType() &&
      a.bunchCrossing() == b.bunchCrossing() &&
      a.storeNumber() == b.storeNumber();
  }
}
