/*----------------------------------------------------------------------
$Id: InputSource.cc,v 1.26 2007/06/22 23:26:33 wmtan Exp $
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

  namespace {
	int const improbable = -65783927;
	std::string const& suffix(int count) {
	  static std::string const st("st");
	  static std::string const nd("nd");
	  static std::string const rd("rd");
	  static std::string const th("th");
	  // *0, *4 - *9 use "th".
	  int lastDigit = count % 10;
	  if (lastDigit >= 4 || lastDigit == 0) return th;
	  // *11, *12, or *13 use "th".
	  if (count % 100 - lastDigit == 10) return th;
	  return (lastDigit == 1 ? st : (lastDigit == 2 ? nd : rd));
        }
	struct do_nothing_deleter {
	  void  operator () (void const*) const {}
	};
	template <typename T>
	boost::shared_ptr<T> createSharedPtrToStatic(T * ptr) {
	  return  boost::shared_ptr<T>(ptr, do_nothing_deleter());
	}
  }
  InputSource::InputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ProductRegistryHelper(),
      maxEvents_(desc.maxEvents_),
      remainingEvents_(maxEvents_),
      readCount_(0),
      unlimited_(maxEvents_ < 0),
      moduleDescription_(desc.moduleDescription_),
      productRegistry_(createSharedPtrToStatic<ProductRegistry const>(desc.productRegistry_)),
      primary_(pset.getParameter<std::string>("@module_label") == std::string("@main_input")) {
    // Secondary input sources currently do not have a product registry.
    if (primary_) {
      assert(desc.productRegistry_ != 0);
    }
    int maxEventsOldStyle = pset.getUntrackedParameter<int>("maxEvents", improbable);
    if (maxEventsOldStyle != improbable) {
      throw edm::Exception(edm::errors::Configuration)
        << "InputSource::InputSource()\n"
	<< "The 'maxEvents' parameter for sources is no longer supported.\n"
        << "Please use instead the process level parameter set\n"
        << "'untracked PSet maxEvents = {untracked int32 input = " << maxEventsOldStyle << "}'\n";
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
      addToRegistry(typeLabelList().begin(), typeLabelList().end(), moduleDescription(), productRegistryUpdate());
    }
  }

  boost::shared_ptr<RunPrincipal>
  InputSource::readRun() {
    // Note: For the moment, we do not support saving and restoring the state of the
    // random number generator if random numbers are generated during processing of runs
    // (e.g. beginRun(), endRun())
    if (remainingEvents_ == 0) {
      return boost::shared_ptr<RunPrincipal>();
    }
    return readRun_();
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  InputSource::readLuminosityBlock(boost::shared_ptr<RunPrincipal> rp) {
    // Note: For the moment, we do not support saving and restoring the state of the
    // random number generator if random numbers are generated during processing of lumi blocks
    // (e.g. beginLuminosityBlock(), endLuminosityBlock())
    if (remainingEvents_ == 0) {
      return boost::shared_ptr<LuminosityBlockPrincipal>();
    }
    return readLuminosityBlock_(rp);
  }

  std::auto_ptr<EventPrincipal>
  InputSource::readEvent(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {

    std::auto_ptr<EventPrincipal> result(0);

    if (remainingEvents_ != 0) {
      preRead();
      result = readEvent_(lbp);
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
			 << suffix(readCount_) << " record. Run " << eventID.run()
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
  InputSource::setLumi(LuminosityBlockNumber_t) {
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
