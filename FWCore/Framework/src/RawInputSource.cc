/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.8 2006/12/28 23:52:02 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    remainingEvents_(maxEvents()),
    runNumber_(RunNumber_t()),
    oldRunNumber_(RunNumber_t()),
    luminosityBlockID_(LuminosityBlockID()),
    oldLuminosityBlockID_(LuminosityBlockID()),
    justBegun_(true),
    ep_(),
    luminosityBlockPrincipal_()
  { }

  RawInputSource::~RawInputSource() {
  }

  void
  RawInputSource::setRun(RunNumber_t r) {
    // Do nothing if the run is not changed.
    if (r != runNumber_) {
      runNumber_ = r;
      luminosityBlockID_ = 1;
    }
  }

  void
  RawInputSource::setLumi(LuminosityBlockID lb) {
    luminosityBlockID_ = lb;
  }

  void
  RawInputSource::finishRun() {
    RunPrincipal & rp =
        const_cast<RunPrincipal &>(luminosityBlockPrincipal_->runPrincipal());
    Run run(rp, moduleDescription());
    endRun(run);
    run.commit_();
  }

  void
  RawInputSource::finishLumi() {
    LuminosityBlockPrincipal & lbp =
        const_cast<LuminosityBlockPrincipal &>(*luminosityBlockPrincipal_);
    LuminosityBlock lb(lbp, moduleDescription());
    endLuminosityBlock(lb);
    lb.commit_();
  }

  void
  RawInputSource::endLumiAndRun() {
    finishLumi();
    finishRun(); 
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::read() {
    if (remainingEvents_ == 0) {
      if (!justBegun_) {
        finishLumi();
	finishRun();
      }
      return std::auto_ptr<EventPrincipal>(0); 
    }
    bool isNewRun = justBegun_ || oldRunNumber_ != runNumber_;
    bool isNewLumi = isNewRun || oldLuminosityBlockID_ != luminosityBlockID_;
    if(!justBegun_ && isNewLumi) {
      finishLumi();
      if (isNewRun) {
	finishRun();
      }
    }
    justBegun_ = false;
    oldLuminosityBlockID_ = luminosityBlockID_;
    oldRunNumber_ = runNumber_;
    if (isNewLumi) {
      if (isNewRun) {
        boost::shared_ptr<RunPrincipal> runPrincipal(
	    new RunPrincipal(runNumber_, productRegistry(), processConfiguration()));
        Run run(*runPrincipal, moduleDescription());
	beginRun(run);
	run.commit_();
        luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal>(
	    new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal, processConfiguration()));
      } else {
        boost::shared_ptr<RunPrincipal const> runPrincipal = luminosityBlockPrincipal_->runPrincipalConstSharedPtr();
        luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal>(
	    new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal, processConfiguration()));
      }
      LuminosityBlockPrincipal & lbp =
         const_cast<LuminosityBlockPrincipal &>(*luminosityBlockPrincipal_);
      LuminosityBlock lb(lbp, moduleDescription());
      beginLuminosityBlock(lb);
      lb.commit_();
    }
    std::auto_ptr<Event> e(readOneEvent());
    if(e.get() != 0) {
      --remainingEvents_;
      e->commit_();
    }
    return ep_;
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(EventID & eventId, Timestamp const& tstamp) {
    eventId = EventID(runNumber_, eventId.event());
    ep_ = std::auto_ptr<EventPrincipal>(
	new EventPrincipal(eventId, Timestamp(tstamp),
	productRegistry(), luminosityBlockPrincipal_, processConfiguration()));
    std::auto_ptr<Event> e(new Event(*ep_, moduleDescription()));
    return e;
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::readIt(EventID const&) {
      throw cms::Exception("LogicError","RawInputSource::read(EventID const& eventID)")
        << "Random access read cannot be used for RawInputSource.\n"
        << "Contact a Framework developer.\n";
  }

  // Not yet implemented
  void
  RawInputSource::skip(int) {
      throw cms::Exception("LogicError","RawInputSource::skip(int offset)")
        << "Random access skip cannot be used for RawInputSource\n"
        << "Contact a Framework developer.\n";
  }

}
