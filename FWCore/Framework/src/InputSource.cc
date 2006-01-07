/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.4 2006/01/07 00:38:14 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  InputSource::InputSource(InputSourceDescription const& desc) :
      preg_(desc.preg_),
      process_(desc.processName_) {
    // Secondary input sources currently do not have a product registry or a process name.
    // So, these asserts are commented out. for now.
    // assert(preg_ != 0);
    // assert(!process_.empty());
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
