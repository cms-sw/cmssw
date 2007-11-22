/*----------------------------------------------------------------------
$Id$
----------------------------------------------------------------------*/

#include "FWCore/Integration/test/TestRunLumiSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edm {
  
  TestRunLumiSource::TestRunLumiSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    runLumiEvent_(pset.getUntrackedParameter<std::vector<int> >("runLumiEvent", std::vector<int>())),
    currentIndex_(0) { 
  }

  TestRunLumiSource::~TestRunLumiSource() {
  }

  boost::shared_ptr<RunPrincipal>
  TestRunLumiSource::readRun_() {
    setRunLumiEvent();
    if (end_) return boost::shared_ptr<RunPrincipal>();
    Timestamp ts = Timestamp(1);  // 1 is just a meaningless number to make it compile for the test
    boost::shared_ptr<RunPrincipal> runPrincipal(
        new RunPrincipal(eventID_.run(), ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));
    return runPrincipal;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  TestRunLumiSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    setRunLumiEvent();
    if (end_) return boost::shared_ptr<LuminosityBlockPrincipal>();
    Timestamp ts = Timestamp(1);

    boost::shared_ptr<RunPrincipal> rp2(
        new RunPrincipal(eventID_.run(), ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));

    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal(
        new LuminosityBlockPrincipal(
	    luminosityBlock_, ts, Timestamp::invalidTimestamp(), productRegistry(), rp2, processConfiguration()));

    return luminosityBlockPrincipal;
  }

  std::auto_ptr<EventPrincipal>
  TestRunLumiSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    setRunLumiEvent();
    if (end_) return std::auto_ptr<EventPrincipal>(0);
    Timestamp ts = Timestamp(1);

    boost::shared_ptr<RunPrincipal> rp2(
        new RunPrincipal(eventID_.run(), ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));

    boost::shared_ptr<LuminosityBlockPrincipal> lbp2(
        new LuminosityBlockPrincipal(
	    luminosityBlock_, ts, Timestamp::invalidTimestamp(), productRegistry(), rp2, processConfiguration()));

    std::auto_ptr<EventPrincipal> result(
	new EventPrincipal(eventID_, ts,
	productRegistry(), lbp2, processConfiguration(), false));
    return result;
  }

  void
  TestRunLumiSource::setRunLumiEvent() {

    int run = 0;
    int lumi = 0;
    int event = 0;

    if (currentIndex_ + 2 < runLumiEvent_.size()) {
      run   = runLumiEvent_[currentIndex_];
      lumi  = runLumiEvent_[currentIndex_ + 1];
      event = runLumiEvent_[currentIndex_ + 2];
      currentIndex_ += 3;
    }

    if (run == 0 && lumi == 0 && event == 0) end_ = true;
    else {
      end_ = false;
      eventID_ = EventID(run, event);
      luminosityBlock_ = lumi;
    }
  }
}

using edm::TestRunLumiSource;
DEFINE_FWK_INPUT_SOURCE(TestRunLumiSource);

