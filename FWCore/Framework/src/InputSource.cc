/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.6 2006/04/04 22:15:22 wmtan Exp $
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
      remainingEvents_(maxEvents_),
      unlimited_(maxEvents_ < 0),
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
  InputSource::readEvent_() {
    if (unlimited_) {
      return read();
    }
    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      result = read();
      if (result.get() != 0) {
        --remainingEvents_;
      }
    }
    return result;
  }

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent() {
    // Do we need any error handling (e.g. exception translation) here?
    std::auto_ptr<EventPrincipal> ep(readEvent_());
    if (ep.get()) {
	ep->addToProcessHistory(process_);
    }
    return ep;
  }


  std::auto_ptr<EventPrincipal>
  InputSource::readEvent_(EventID const& eventID) {
    if (unlimited_) {
      return readIt(eventID);
    }
    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      result = readIt(eventID);
      if (result.get() != 0) {
        --remainingEvents_;
      }
    }
    return result;
  }

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent(EventID const& eventID) {
    // Do we need any error handling (e.g. exception translation) here?
    std::auto_ptr<EventPrincipal> ep(readEvent_(eventID));
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
