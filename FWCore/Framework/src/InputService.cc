/*----------------------------------------------------------------------
$Id: InputService.cc,v 1.4 2005/07/14 22:50:53 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  InputService::InputService(InputServiceDescription const& desc) :
      process_(desc.process_name),
      preg_(desc.preg_) { 
    assert(!process_.empty());
    assert(preg_ != 0);
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
