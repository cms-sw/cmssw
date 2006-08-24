#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: EventAux.cc,v 1.5 2006/07/21 21:56:41 wmtan Exp $

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

  void
  EventAux::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
    os << "LuminosityBlockID = " << luminosityBlockID_ << std::endl;
  }
}
