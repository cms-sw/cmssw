#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include <string>

static void ensurePluginManager() {
  static bool configured = false;
  if (!configured) {
    if (!edmplugin::PluginManager::isAvailable()) {
      edmplugin::PluginManager::configure(edmplugin::standard::config());
    }
    configured = true;
  }
}

// Test double for TritonClient to observe updateServer calls without framework/services
class TestTritonClient : public TritonClient {
public:
  TestTritonClient() : TritonClient() {}

  void updateServer(const std::string& serverName) override { lastUpdatedServerName = serverName; }

  const std::string& lastServerName() const { return lastUpdatedServerName; }

protected:
  void evaluate() override {}

private:
  std::string lastUpdatedServerName;
};

TEST_CASE("RetryActionDiffServer handles a missing TritonService gracefully", "[RetryActionDiffServer]") {
  // Outside the full framework there is no ServiceRegistry, so TritonClient::service()
  // cannot resolve a TritonService and querying for an alternative server fails. This
  // exercises that retry() catches that failure without throwing, does not call
  // updateServer (since no alternative server could be determined), and still disarms
  // itself after the one allowed attempt.
  ensurePluginManager();
  edm::ParameterSet empty;
  TestTritonClient client;

  auto action =
      RetryActionFactory::get()->create("RetryActionDiffServer", empty, static_cast<SonicClientBase*>(&client));

  // start should arm the action
  action->start();
  REQUIRE(action->shouldRetry());

  // retry should not throw despite the missing TritonService, and should not call
  // updateServer since no alternative server could be resolved
  REQUIRE_NOTHROW(action->retry());
  REQUIRE(client.lastServerName().empty());

  // one-time use: retry disarms itself after the first attempt
  REQUIRE_FALSE(action->shouldRetry());

  // second retry without re-arming should still be a no-op: lastServerName unchanged
  action->retry();
  REQUIRE(client.lastServerName().empty());
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
  ensurePluginManager();
  edm::ParameterSet empty;
  ThrowingTritonClient client;
  auto action =
      RetryActionFactory::get()->create("RetryActionDiffServer", empty, static_cast<SonicClientBase*>(&client));
  action->start();

  // Should not throw despite client throwing internally; action disarms afterward
  REQUIRE_NOTHROW(action->retry());
}
