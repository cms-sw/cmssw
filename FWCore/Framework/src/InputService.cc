/*----------------------------------------------------------------------
$Id: InputService.cc,v 1.6 2005/07/27 23:38:22 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  InputService::InputService(InputServiceDescription const& desc) :
      preg_(desc.preg_),
      process_(desc.processName_) {
    assert(preg_ != 0);
    assert(!process_.empty());
  }

  InputService::~InputService() {}

  std::auto_ptr<EventPrincipal>
  InputService::readEvent() {
    // Do we need any error handling (e.g. exception translation)
    // here?
    std::auto_ptr<EventPrincipal> ep(this->read());
    if (ep.get()) {
	ep->addToProcessHistory(process_);
    }
    return ep;
  }
}
