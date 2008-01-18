/*----------------------------------------------------------------------
$Id: TestRunLumiSource.cc,v 1.4 2007/12/31 22:43:58 wmtan Exp $
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
    currentIndex_(0),
    firstTime_(true) { 
  }

  TestRunLumiSource::~TestRunLumiSource() {
  }

  boost::shared_ptr<RunPrincipal>
  TestRunLumiSource::readRun_() {
    unsigned int run = runLumiEvent_[currentIndex_];
    Timestamp ts = Timestamp(1);  // 1 is just a meaningless number to make it compile for the test

    boost::shared_ptr<RunPrincipal> runPrincipal(
        new RunPrincipal(run, ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));
    currentIndex_ += 3;
    return runPrincipal;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  TestRunLumiSource::readLuminosityBlock_() {
    unsigned int run = runLumiEvent_[currentIndex_];
    unsigned int lumi = runLumiEvent_[currentIndex_ + 1];
    Timestamp ts = Timestamp(1);

    boost::shared_ptr<RunPrincipal> rp2(
        new RunPrincipal(run, ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));

    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal(
        new LuminosityBlockPrincipal(lumi,
	    ts, Timestamp::invalidTimestamp(), productRegistry(), rp2, processConfiguration()));

    currentIndex_ += 3;
    return luminosityBlockPrincipal;
  }

  std::auto_ptr<EventPrincipal>
  TestRunLumiSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    unsigned int run = runLumiEvent_[currentIndex_];
    unsigned int lumi = runLumiEvent_[currentIndex_ + 1];
    unsigned int event = runLumiEvent_[currentIndex_ + 2];
    Timestamp ts = Timestamp(1);

    boost::shared_ptr<RunPrincipal> rp2(
        new RunPrincipal(run, ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));

    boost::shared_ptr<LuminosityBlockPrincipal> lbp2(
        new LuminosityBlockPrincipal(lumi,
	    ts, Timestamp::invalidTimestamp(), productRegistry(), rp2, processConfiguration()));

    EventID id(run, event);
    currentIndex_ += 3;
    std::auto_ptr<EventPrincipal> result(
	new EventPrincipal(id, processGUID(), ts, productRegistry(), lbp2, processConfiguration(), false));
    return result;
  }

  InputSource::ItemType
  TestRunLumiSource::getNextItemType() {
    if (firstTime_) {
      firstTime_ = false;
      return InputSource::IsFile;
    }
    if (currentIndex_ + 2 >= runLumiEvent_.size()) {
      return InputSource::IsStop;
    }
    if (runLumiEvent_[currentIndex_] == 0) {
      return InputSource::IsStop;
    }
    ItemType oldState = state();
    if (oldState == IsInvalid) return InputSource::IsFile;
    if (runLumiEvent_[currentIndex_ + 1] == 0) {
      return InputSource::IsRun;
    }
    if (runLumiEvent_[currentIndex_ + 2] == 0) {
      return InputSource::IsLumi;
    }
    return InputSource::IsEvent;
  }
}

using edm::TestRunLumiSource;
DEFINE_FWK_INPUT_SOURCE(TestRunLumiSource);

