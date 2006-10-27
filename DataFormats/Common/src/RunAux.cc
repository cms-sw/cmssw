#include "DataFormats/Common/interface/RunAux.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: RunAux.cc,v 1.7 2006/10/07 03:50:05 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  RunAux::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
  }
}
