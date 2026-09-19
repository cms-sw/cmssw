#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

// Custom deleter implementation
void SonicClientBase::RetryDeleter::operator()(RetryActionBase* ptr) const { delete ptr; }

SonicClientBase::SonicClientBase(const edm::ParameterSet& params,
                                 const std::string& debugName,
                                 const std::string& clientName)
    : debugName_(debugName),
      clientName_(clientName),
      fullDebugName_(debugName_),
      userMode_(params.getParameter<std::string>("mode")) {
  if (!clientName_.empty())
    fullDebugName_ += ":" + clientName_;

  const auto& retryPSetList = params.getParameter<std::vector<edm::ParameterSet>>("Retry");

  for (const auto& retryPSet : retryPSetList) {
    const std::string& actionType = retryPSet.getParameter<std::string>("retryType");

    auto retryAction = RetryActionFactory::get()->create(actionType, retryPSet, this);
    if (retryAction) {
      //Convert to  RetryActionPtr Type from raw pointer of retryAction
      retryActions_.emplace_back(RetryActionPtr(retryAction.release()));
    } else {
      throw cms::Exception("Configuration")
          << "Unknown Retry type " << actionType << " for SonicClient: " << fullDebugName_;
    }
  }

  setUserMode(userMode_);
}
void SonicClientBase::setUserMode(const std::string& userMode) {
  if (userMode == "Sync")
    setMode(SonicMode::Sync);
  else if (userMode == "Async")
    setMode(SonicMode::Async);
  else if (userMode == "PseudoAsync")
    setMode(SonicMode::PseudoAsync);
  else if (userMode.empty())
    setMode(SonicMode::PseudoAsync);
  else
    throw cms::Exception("Configuration") << "Unknown mode for SonicClient: " << userMode;
}
void SonicClientBase::setMode(SonicMode mode) {
  if (dispatcher_ and mode_ == mode)
    return;
  mode_ = mode;

  //get correct dispatcher for mode
  if (mode_ == SonicMode::Sync or mode_ == SonicMode::Async)
    dispatcher_ = std::make_unique<SonicDispatcher>(this);
  else if (mode_ == SonicMode::PseudoAsync)
    dispatcher_ = std::make_unique<SonicDispatcherPseudoAsync>(this);
}

void SonicClientBase::start(edm::WaitingTaskWithArenaHolder holder) {
  start();
  holder_ = std::move(holder);
}

void SonicClientBase::start() {
  totalTries_ = 0;
  // initialize all actions
  for (auto& action : retryActions_) {
    action->start();
  }
}

void SonicClientBase::finish(bool success, std::exception_ptr eptr) {
  //retries are only allowed if no exception was raised
  if (!success and !eptr) {
    ++totalTries_;
    edm::LogInfo("SonicClientBase") << "finish: failed after total tries of " << totalTries_;
    for (const auto& action : retryActions_) {
      if (action->shouldRetry()) {
        edm::LogInfo("SonicClientBase") << "Calling retry()";
        // retry() must trigger eval() or finish()
        action->retry();
        return;
      }
    }
    //prepare an exception if no more retry actions left
    edm::LogInfo("SonicClientBase") << "SonicCallFailed: call failed, no retry actions available after " << totalTries_
                                    << " tries.";
    edm::Exception ex(edm::errors::ExternalFailure);
    ex << "SonicCallFailed: call failed, no retry actions available after " << totalTries_ << " tries.";
    eptr = make_exception_ptr(ex);
  }
  if (holder_) {
    holder_->doneWaiting(eptr);
    holder_.reset();
  } else if (eptr)
    std::rethrow_exception(eptr);

  //reset client data now (usually done at end of produce())
  if (eptr)
    reset();
}

void SonicClientBase::fillBasePSetDescription(edm::ParameterSetDescription& desc, const std::string& defaultRetryType) {
  //restrict allowed values
  desc.ifValue(edm::ParameterDescription<std::string>("mode", "", true),
               edm::allowedValues<std::string>("Sync", "Async", "PseudoAsync", ""));

  // Defines the structure of each entry in the VPSet
  edm::ParameterSetDescription retryDesc;
  retryDesc.add<std::string>("retryType", defaultRetryType);
  retryDesc.addUntracked<unsigned>("allowedTries", 0);  //used by RetrySameServerAction only

  // Define a default retry action
  edm::ParameterSet defaultRetry;
  defaultRetry.addParameter<std::string>("retryType", defaultRetryType);
  defaultRetry.addUntrackedParameter<unsigned>("allowedTries", 0);

  // Add the VPSet with the default retry action
  desc.addVPSet("Retry", retryDesc, {defaultRetry});

  desc.add("sonicClientBase", desc);
  desc.addUntracked<bool>("verbose", false);
}
