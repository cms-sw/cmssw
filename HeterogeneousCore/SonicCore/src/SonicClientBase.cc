#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

SonicClientBase::SonicClientBase(const edm::ParameterSet& params)
    : allowedTries_(params.getUntrackedParameter<unsigned>("allowedTries", 0)) {
  std::string modeName(params.getParameter<std::string>("mode"));
  if (modeName == "Sync")
    mode_ = SonicMode::Sync;
  else if (modeName == "Async")
    mode_ = SonicMode::Async;
  else if (modeName == "PseudoAsync")
    mode_ = SonicMode::PseudoAsync;
  else
    throw cms::Exception("Configuration") << "Unknown mode for SonicClient: " << modeName;

  //get correct dispatcher for mode
  if (mode_ == SonicMode::Sync or mode_ == SonicMode::Async)
    dispatcher_ = std::make_unique<SonicDispatcher>(this);
  else if (mode_ == SonicMode::PseudoAsync)
    dispatcher_ = std::make_unique<SonicDispatcherPseudoAsync>(this);
}

void SonicClientBase::setDebugName(const std::string& debugName) {
  debugName_ = debugName;
  fullDebugName_ = debugName_;
  if (!clientName_.empty())
    fullDebugName_ += ":" + clientName_;
}

void SonicClientBase::start(edm::WaitingTaskWithArenaHolder holder) {
  holder_ = std::move(holder);
  tries_ = 0;
  if (!debugName_.empty())
    t0_ = std::chrono::high_resolution_clock::now();
}

void SonicClientBase::finish(bool success, std::exception_ptr eptr) {
  //retries are only allowed if no exception was raised
  if (!success and !eptr) {
    ++tries_;
    //if max retries has not been exceeded, call evaluate again
    if (tries_ < allowedTries_) {
      evaluate();
      //avoid calling doneWaiting() twice
      return;
    }
    //prepare an exception if exceeded
    else {
      cms::Exception ex("SonicCallFailed");
      ex << "call failed after max " << tries_ << " tries";
      eptr = make_exception_ptr(ex);
    }
  }
  if (!debugName_.empty()) {
    auto t1 = std::chrono::high_resolution_clock::now();
    edm::LogInfo(fullDebugName_) << "Client time: "
                                 << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0_).count();
  }
  holder_.doneWaiting(eptr);
}

void SonicClientBase::fillBasePSetDescription(edm::ParameterSetDescription& desc, bool allowRetry) {
  //restrict allowed values
  desc.ifValue(edm::ParameterDescription<std::string>("mode", "PseudoAsync", true),
               edm::allowedValues<std::string>("Sync", "Async", "PseudoAsync"));
  if (allowRetry)
    desc.addUntracked<unsigned>("allowedTries", 0);
}
