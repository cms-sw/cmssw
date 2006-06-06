/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.11 2006/05/02 02:15:08 wmtan Exp $
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
      module_(desc.module_),
      preg_(desc.preg_) {
    // Secondary input sources currently do not have a product registry.
    // So, this assert is commented out. for now.
    // assert(preg_ != 0);
  }

  InputSource::~InputSource() {}

  void
  InputSource::beginJob(EventSetup const&) { }

  void
  InputSource::endJob() { }

  void
  InputSource::registerProducts() {
    if (!typeLabelList().empty()) {
      addToRegistry(typeLabelList().begin(), typeLabelList().end(), module(), *preg_);
    }
  }

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
	ep->addToProcessHistory(module().processName_);
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
	ep->addToProcessHistory(module().processName_);
    }
    return ep;
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

  std::auto_ptr<EventPrincipal>
  InputSource::readIt(EventID const&) {
      throw edm::Exception(edm::errors::LogicError)
        << "InputSource::readIt()\n"
        << "Random access is not implemented for this type of Input Source\n"
        << "Contact a Framework Developer\n";
  }

  void
  InputSource::setRun(RunNumber_t) {
      throw edm::Exception(edm::errors::LogicError)
        << "InputSource::setRun()\n"
        << "Run number cannot be modified for this type of Input Source\n"
        << "Contact a Framework Developer\n";
  }

  void
  InputSource::skip(int) {
      throw edm::Exception(edm::errors::LogicError)
        << "InputSource::skip()\n"
        << "Random access is not implemented for this type of Input Source\n"
        << "Contact a Framework Developer\n";
  }

  void
  InputSource::rewind_() {
      throw edm::Exception(edm::errors::LogicError)
        << "InputSource::rewind()\n"
        << "Rewind is not implemented for this type of Input Source\n"
        << "Contact a Framework Developer\n";
  }

}
