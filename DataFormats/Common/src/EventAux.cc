#include "DataFormats/Common/interface/EventAux.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: EventAux.cc,v 1.6 2006/08/24 22:15:44 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAux::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
    os << "LuminosityBlockID = " << luminosityBlockID_ << std::endl;
  }
}
