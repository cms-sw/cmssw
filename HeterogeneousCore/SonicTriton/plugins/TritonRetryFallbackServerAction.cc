// TritonRetryFallbackServerAction: last-resort retry action for TritonClient.
//
// When all other retry actions have been exhausted, this action loads the
// client's model onto the fallback (local) Triton server and re-runs
// inference there.  It fires at most once per inference call.
//
// Usage add to the Retry VPSet *after* all other retry actions:
//   cms.PSet(retryType = cms.string("TritonRetryFallbackServerAction"))
//
// Requirements:
//   - TritonService fallback must be enabled in the job configuration.
//   - The model must have a modelConfigPath / repository path known to
//     TritonService so it can be loaded dynamically.

#include "HeterogeneousCore/SonicTriton/interface/TritonRetryActionBase.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class TritonRetryFallbackServerAction : public TritonRetryActionBase {
public:
  TritonRetryFallbackServerAction(const edm::ParameterSet& conf, SonicClientBase& client);
  ~TritonRetryFallbackServerAction() override = default;

  void retry() override;
  void start() override;
};

TritonRetryFallbackServerAction::TritonRetryFallbackServerAction(const edm::ParameterSet& conf, SonicClientBase& client)
    : TritonRetryActionBase(conf, client) {}

void TritonRetryFallbackServerAction::start() { this->shouldRetry_ = true; }

void TritonRetryFallbackServerAction::retry() {
  // Allow only one fallback attempt per inference call.
  shouldRetry_ = false;

  CMS_SA_ALLOW try {
    // Start the fallback server (idempotent), load the model, and point
    // the client's gRPC connection at the fallback URL.
    tritonClient().switchToFallback();
    // Re-run the inference on the fallback server.
    eval();
  } catch (...) {
    // Non-retryable: propagate the exception so the job fails cleanly.
    finish(false);
  }
}
DEFINE_RETRY_ACTION(TritonRetryFallbackServerAction);
