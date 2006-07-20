#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"

/*----------------------------------------------------------------------

$Id: EventAux.cc,v 1.3 2006/07/14 23:00:55 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAux::init() const {
    if (processHistoryPtr_.get() == 0) {
      processHistoryPtr_ = boost::shared_ptr<ProcessHistory>(new ProcessHistory);
      if (processHistoryID_ != ProcessHistoryID()) {
        assert(ProcessHistoryRegistry::instance()->size());
        bool found = ProcessHistoryRegistry::instance()->getMapped(processHistoryID_, *processHistoryPtr_);
        assert(found);
      }
    }
  }
}
