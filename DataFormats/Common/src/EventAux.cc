#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/ProcessHistoryRegistry.h"

/*----------------------------------------------------------------------

$Id: EventAux.cc,v 1.2 2006/07/06 18:34:06 wmtan Exp $

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
