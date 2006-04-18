/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.8 2006/04/18 21:55:21 lsexton Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


namespace edm {

  InputSource::InputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ProductRegistryHelper(),
      maxEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
      remainingEvents_(maxEvents_),
      readCount_(0),
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

    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      result = read();
      if (result.get() != 0) {
        if (!unlimited_) --remainingEvents_;
	++readCount_;
	issueReports(result->id());
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

    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      result = readIt(eventID);
      if (result.get() != 0) {
        if (!unlimited_) --remainingEvents_;
	++readCount_;
	issueReports(result->id());
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
  void
  InputSource::issueReports(EventID const& eventID) {
    LogInfo("FwkReport") << "Begin processing the " << readCount_
			 << "th record. Run " <<  eventID.run()
			 << ", Event " << eventID.event();
      // At some point we may want to initiate checkpointing here
  }
}
