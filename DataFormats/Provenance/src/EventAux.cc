#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"

/*----------------------------------------------------------------------

$Id: EventAux.cc,v 1.1 2007/03/04 04:48:10 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void conversion(EventAux const& from, EventAuxiliary & to) {
    to.processHistoryID_ = from.processHistoryID_;
    to.id_ = from.id_;
    to.time_ = from.time_;
    to.luminosityBlock_ = from.luminosityBlockID_;
  }
}
