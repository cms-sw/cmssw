#ifndef HeterogeneousCore_SonicTriton_RetryActionDiffServer_h
#define HeterogeneousCore_SonicTriton_RetryActionDiffServer_h

#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"

/**
 * @class RetryActionDiffServer
 * @brief A concrete implementation of RetryActionBase that attempts to retry an inference
 * request on a different Triton server.
 *
 * This class provides a fallback mechanism. If an initial inference request fails
 * (e.g., due to server unavailability or a model-specific error), this action will be
 * triggered. It queries the central TritonService to select an alternative server (e.g.,
 * the fallback server when available) and instructs the TritonClient to reconnect to
 * that server for the retry attempt. This action is designed for one-time use per
 * inference call; after the retry attempt, it disables itself until the next `start()`
 * call.
 */

class RetryActionDiffServer : public RetryActionBase {
public:
  RetryActionDiffServer(const edm::ParameterSet& conf, SonicClientBase* client);
  ~RetryActionDiffServer() override = default;

  void retry() override;
  void start() override;

private:
  unsigned tries_;
};

#endif
