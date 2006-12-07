#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: LuminosityBlockAux.cc,v 1.1 2006/10/27 20:57:49 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  LuminosityBlockAux::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
  }
}
