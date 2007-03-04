#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: LuminosityBlockAuxiliary.cc,v 1.2 2006/12/07 23:48:56 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  LuminosityBlockAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
  }
}
