/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.18 2006/12/21 00:05:36 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert> 
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

namespace edm {

  InputSource::InputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ProductRegistryHelper(),
      maxEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
      remainingEvents_(maxEvents_),
      readCount_(0),
      unlimited_(maxEvents_ < 0),
      isDesc_(desc),
      primary_(pset.getParameter<std::string>("@module_label") == std::string("@main_input")) {
    // Secondary input sources currently do not have a product registry.
    if (primary_) {
      assert(isDesc_.productRegistry_ != 0);
    }
  }

  InputSource::~InputSource() {}

  void
  InputSource::doBeginJob(EventSetup const& c) {
    beginJob(c);
  }

  void
  InputSource::doEndJob() {
    endJob();
  }

  void
  InputSource::registerProducts() {
    if (!typeLabelList().empty()) {
      addToRegistry(typeLabelList().begin(), typeLabelList().end(), moduleDescription(), productRegistry());
    }
  }

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent() {

    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      preRead();
      result = read();
      if (result.get() != 0) {
        Event event(*result, moduleDescription());
        postRead(event);
        if (!unlimited_) --remainingEvents_;
	++readCount_;
	issueReports(result->id());
      }
    } else {
      endLumiAndRun();
    }
    return result;
  }

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent(EventID const& eventID) {

    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      preRead();
      result = readIt(eventID);
      if (result.get() != 0) {
        Event event(*result, moduleDescription());
        postRead(event);
        if (!unlimited_) --remainingEvents_;
	++readCount_;
	issueReports(result->id());
      }
    }
    return result;
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
  InputSource::setLumi(LuminosityBlockID) {
      throw edm::Exception(edm::errors::LogicError)
        << "InputSource::setLumi()\n"
        << "Luminosity Block ID  cannot be modified for this type of Input Source\n"
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

  void 
  InputSource::preRead() {

    if (primary()) {
      Service<RandomNumberGenerator> rng;
      if (rng.isAvailable()) {
        rng->snapShot();
      }
    }
  }

  void 
  InputSource::postRead(Event& event) {

    if (primary()) {
      Service<RandomNumberGenerator> rng;
      if (rng.isAvailable()) {
        rng->restoreState(event);
      }
    }
  }
}
