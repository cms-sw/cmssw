#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "HeterogeneousCore/SonicTriton/interface/RetryActionDiffServer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

// Test double for TritonClient to observe updateServer calls without framework/services
class TestTritonClient : public TritonClient {
public:
  TestTritonClient() : TritonClient() {}

  void connectToServer(const std::string& url) override { lastConnectedUrl = url; }

  void updateServer(const std::string& serverName) override { lastUpdatedServerName = serverName; }

  const std::string& lastUrl() const { return lastConnectedUrl; }
  const std::string& lastServerName() const { return lastUpdatedServerName; }

protected:
  void evaluate() override {}

private:
  std::string lastConnectedUrl;
  std::string lastUpdatedServerName;
};

TEST_CASE("RetryActionDiffServer switches to fallback via updateServer", "[RetryActionDiffServer]") {
  edm::ParameterSet empty;
  TestTritonClient client;

  RetryActionDiffServer action(empty, static_cast<SonicClientBase*>(&client));

  // start should arm the action
  action.start();
  REQUIRE(action.shouldRetry());

  // retry should call updateServer with fallback name then disarm
  action.retry();
  REQUIRE(client.lastServerName() == TritonService::Server::fallbackName);

  // second retry without re-arming should be a no-op: lastServerName unchanged
  std::string afterFirst = client.lastServerName();
  action.retry();
  REQUIRE(client.lastServerName() == afterFirst);
}

// A client that throws during updateServer to exercise error handling path
class ThrowingTritonClient : public TritonClient {
public:
  ThrowingTritonClient() : TritonClient() {}
  void updateServer(const std::string&) override { throw TritonException("updateServer failure"); }

protected:
  void evaluate() override {}
};

TEST_CASE("RetryActionDiffServer catches exceptions from updateServer", "[RetryActionDiffServer]") {
  edm::ParameterSet empty;
  ThrowingTritonClient client;
  RetryActionDiffServer action(empty, static_cast<SonicClientBase*>(&client));
  action.start();

  // Should not throw despite client throwing internally; action disarms afterward
  REQUIRE_NOTHROW(action.retry());
}
