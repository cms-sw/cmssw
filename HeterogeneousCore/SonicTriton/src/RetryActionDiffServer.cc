#include "HeterogeneousCore/SonicTriton/interface/RetryActionDiffServer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

RetryActionDiffServer::RetryActionDiffServer(const edm::ParameterSet& conf, SonicClientBase* client)
    : RetryActionBase(conf, client) {}

void RetryActionDiffServer::start() {
  this->shouldRetry_ = true;
  tries_ = 0;
}

void RetryActionDiffServer::retry() {
  ++tries_;
  if (tries_ >= 1) {
    shouldRetry_ = false;  // Flip flag when max retries are reached. Allow 1 try for now.
    edm::LogInfo("RetryDiffServerAction") << "Max retry attempts reached. No further retries.";
  }
  try {
    auto* tritonClient = static_cast<TritonClient*>(client_);
    edm::LogInfo("RetryActionDiffServer") << "Asking for a different server from TritonService";
    auto ts = tritonClient->service();

    // First, try to find another remote server
    auto bestServerName = ts->getBestServer(tritonClient->modelName(), tritonClient->serverName());

    if (bestServerName) {
      edm::LogInfo("RetryActionDiffServer") << "Got best server from service ";
      tritonClient->updateServer(*bestServerName);
      edm::LogInfo("RetryActionDiffServer") << "eval() with new server";
      eval();
      return;
    } else {
      edm::LogWarning("RetryActionDiffServer")
          << "No alternative server found for model " << tritonClient->modelName() << ". Now call client->finish()";
      finish(false);
      return;
    }
  } catch (TritonException& e) {
    e.convertToWarning();
  } catch (std::exception& e) {
    edm::LogError("RetryActionDiffServer") << "Failed to retry with alternative server: " << e.what();
  } catch (...) {
    edm::LogError("RetryActionDiffServer: UnknownFailure") << "An unknown exception was thrown";
  }
}

DEFINE_RETRY_ACTION(RetryActionDiffServer);
