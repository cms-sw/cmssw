/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Integration/test/TestRunLumiSource.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
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

  boost::shared_ptr<RunAuxiliary>
  TestRunLumiSource::readRunAuxiliary_() {
    unsigned int run = runLumiEvent_[currentIndex_];
    Timestamp ts = Timestamp(1);  // 1 is just a meaningless number to make it compile for the test
    currentIndex_ += 3;
    return boost::shared_ptr<RunAuxiliary>(new RunAuxiliary(run, ts, Timestamp::invalidTimestamp()));
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  TestRunLumiSource::readLuminosityBlockAuxiliary_() {
    unsigned int run = runLumiEvent_[currentIndex_];
    unsigned int lumi = runLumiEvent_[currentIndex_ + 1];
    Timestamp ts = Timestamp(1);
    currentIndex_ += 3;
    return boost::shared_ptr<LuminosityBlockAuxiliary>(new LuminosityBlockAuxiliary(run, lumi, ts, Timestamp::invalidTimestamp()));
  }

  EventPrincipal*
  TestRunLumiSource::readEvent_() {
    EventSourceSentry(*this);
    unsigned int run = runLumiEvent_[currentIndex_];
    unsigned int lumi = runLumiEvent_[currentIndex_ + 1];
    unsigned int event = runLumiEvent_[currentIndex_ + 2];
    Timestamp ts = Timestamp(1);

    boost::shared_ptr<RunAuxiliary> runAux(new RunAuxiliary(run, ts, Timestamp::invalidTimestamp()));
    boost::shared_ptr<RunPrincipal> rp2(
        new RunPrincipal(runAux, productRegistry(), processConfiguration()));

    boost::shared_ptr<LuminosityBlockAuxiliary> lumiAux(
	new LuminosityBlockAuxiliary(rp2->run(), lumi, ts, Timestamp::invalidTimestamp()));
    boost::shared_ptr<LuminosityBlockPrincipal> lbp2(
        new LuminosityBlockPrincipal(lumiAux, productRegistry(), processConfiguration(), rp2));

    EventID id(run, lbp2->luminosityBlock(), event);
    currentIndex_ += 3;
    EventAuxiliary eventAux(id, processGUID(), ts, false);
    EventPrincipal* result(new EventPrincipal(productRegistry(), processConfiguration()));
    result->fillEventPrincipal(eventAux, lbp2);
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

