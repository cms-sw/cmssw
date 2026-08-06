// RetryFallbackServerAction: last-resort retry action for TritonClient.
//
// When all other retry actions have been exhausted, this action loads the
// client's model onto the fallback (local) Triton server and re-runs
// inference there.  It fires at most once per inference call.
//
// Usage add to the Retry VPSet *after* all other retry actions:
//   cms.PSet(retryType = cms.string("RetryFallbackServerAction"))
//
// Requirements:
//   - TritonService fallback must be enabled in the job configuration.
//   - The model must have a modelConfigPath / repository path known to
//     TritonService so it can be loaded dynamically.

#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class RetryFallbackServerAction : public RetryActionBase {
public:
  RetryFallbackServerAction(const edm::ParameterSet& conf, SonicClientBase* client);
  ~RetryFallbackServerAction() override = default;

  void retry() override;
  void start() override;

private:
  unsigned tries_;
};

RetryFallbackServerAction::RetryFallbackServerAction(const edm::ParameterSet& conf, SonicClientBase* client)
    : RetryActionBase(conf, client) {}

void RetryFallbackServerAction::start() {
  this->shouldRetry_ = true;
  tries_ = 0;
}

void RetryFallbackServerAction::retry() {
  // Allow only one fallback attempt per inference call.
  shouldRetry_ = false;

  auto* tc = dynamic_cast<TritonClient*>(client_);
  if (!tc) {
    // Should never happen in a correctly configured job.
    edm::LogWarning("RetryFallbackServerAction")
        << "client_ is not a TritonClient — cannot redirect to fallback server";
    finish(false);
    return;
  }

  CMS_SA_ALLOW try {
    // Start the fallback server (idempotent), load the model, and point
    // the client's gRPC connection at the fallback URL.
    tc->switchToFallback();
    // Re-run the inference on the fallback server.
    eval();
  } catch (...) {
    // Non-retryable: propagate the exception so the job fails cleanly.
    finish(false);
  }
}
DEFINE_RETRY_ACTION(RetryFallbackServerAction);
