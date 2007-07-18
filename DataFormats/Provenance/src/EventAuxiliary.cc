#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: EventAuxiliary.cc,v 1.2 2007/07/18 13:22:38 marafino Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
    os << "LuminosityBlockNumber_t = " << luminosityBlock_ << std::endl;
  }
}
