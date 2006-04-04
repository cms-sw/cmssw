/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.5 2006/01/07 20:41:12 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  InputSource::InputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ProductRegistryHelper(),
      maxEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
      module_(),
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
    std::auto_ptr<EventPrincipal> ep(this->readIt(eventID));
    if (ep.get()) {
	ep->addToProcessHistory(process_);
    }
    return ep;
  }

  void
  InputSource::addToRegistry(ModuleDescription const& md) {
    module_ = md;
    if (!typeLabelList().empty()) {
      ProductRegistryHelper::addToRegistry(typeLabelList().begin(), typeLabelList().end(), md, productRegistry());
    }
  }

  void
  InputSource::skipEvents(int offset) {
    this->skip(offset);
  }
}
