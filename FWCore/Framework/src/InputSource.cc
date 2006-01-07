/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.3 2005/12/28 00:48:40 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  InputSource::InputSource(InputSourceDescription const& desc) :
      preg_(desc.preg_),
      process_(desc.processName_) {
    assert(preg_ != 0);
    assert(!process_.empty());
  }

  InputSource::~InputSource() {}

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent() {
    // Do we need any error handling (e.g. exception translation) here?
    std::auto_ptr<EventPrincipal> ep(this->read());
    if (ep.get()) {
	ep->addToProcessHistory(process_);
    }
    return ep;
  }

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent(EventID const& eventID) {
    // Do we need any error handling (e.g. exception translation) here?
    std::auto_ptr<EventPrincipal> ep(this->read(eventID));
    if (ep.get()) {
	ep->addToProcessHistory(process_);
    }
    return ep;
  }

  void
  InputSource::addToReg(ModuleDescription const&) {}

  void
  InputSource::skipEvents(int offset) {
    this->skip(offset);
  }
}
