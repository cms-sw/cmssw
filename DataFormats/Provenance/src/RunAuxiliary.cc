#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: RunAuxiliary.cc,v 1.2 2006/12/07 23:48:56 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  RunAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
  }
}
