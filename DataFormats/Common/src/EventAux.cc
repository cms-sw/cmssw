#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/ProcessHistoryRegistry.h"

/*----------------------------------------------------------------------

$Id: EventAux.cc,v 1.1.2.4 2006/07/04 13:56:44 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAux::init() const {
    if (processHistoryPtr_.get() == 0) {
      processHistoryPtr_ = boost::shared_ptr<ProcessHistory>(new ProcessHistory);
      if (processHistoryID_ != ProcessHistoryID()) {
        bool found = ProcessHistoryRegistry::instance()->getMapped(processHistoryID_, *processHistoryPtr_);
        assert(found);
      }
    }
  }
}
