#include "HeterogeneousCore/SonicCore/interface/SonicRetryActionBase.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

/**
 * @class TritonRetryActionDifferentServer
 * @brief A concrete implementation of SonicRetryActionBase that attempts to retry an inference
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

class TritonRetryActionDifferentServer : public SonicRetryActionBase {
public:
  TritonRetryActionDifferentServer(const edm::ParameterSet& conf, SonicClientBase* client);
  ~TritonRetryActionDifferentServer() override = default;

  void retry() override;
  void start() override;

private:
  unsigned tries_;
};

TritonRetryActionDifferentServer::TritonRetryActionDifferentServer(const edm::ParameterSet& conf,
                                                                   SonicClientBase* client)
    : SonicRetryActionBase(conf, client) {}

void TritonRetryActionDifferentServer::start() {
  this->shouldRetry_ = true;
  tries_ = 0;
}

void TritonRetryActionDifferentServer::retry() {
  ++tries_;
  if (tries_ >= 1) {
    shouldRetry_ = false;  // Flip flag when max retries are reached. Allow 1 try for now.
    edm::LogInfo("TritonRetryActionDifferentServer") << "Max retry attempts reached. No further retries.";
  }
  try {
    auto* tritonClient = static_cast<TritonClient*>(client_);
    edm::LogInfo("TritonRetryActionDifferentServer") << "Asking for a different server from TritonService";
    auto ts = tritonClient->service();

    // First, try to find another remote server
    auto bestServerName = ts->getBestServer(tritonClient->modelName(), tritonClient->serverName());

    if (bestServerName) {
      edm::LogInfo("TritonRetryActionDifferentServer") << "Got best server from service";
      tritonClient->updateServer(*bestServerName);
      edm::LogInfo("TritonRetryActionDifferentServer") << "eval() with new server";
      eval();
      return;
    } else {
      edm::LogWarning("TritonRetryActionDifferentServer")
          << "No alternative server found for model " << tritonClient->modelName();
      finish(false);
      return;
    }
  } catch (TritonException& e) {
    e.convertToWarning();
  } catch (std::exception& e) {
    edm::LogError("TritonRetryActionDifferentServer") << "Failed to retry with alternative server: " << e.what();
  } catch (...) {
    edm::LogError("TritonRetryActionDifferentServer: UnknownFailure") << "An unknown exception was thrown";
  }
}

DEFINE_RETRY_ACTION(TritonRetryActionDifferentServer);
