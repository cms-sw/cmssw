#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicCore/interface/SonicRetryActionBase.h"

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

  // start() is protected in the base class; make it callable from the test.
  using TritonClient::start;

protected:
  void evaluate() override {}

private:
  std::string lastUpdatedServerName;
};

TEST_CASE("TritonRetryActionDifferentServer handles a missing TritonService gracefully",
          "[TritonRetryActionDifferentServer]") {
  // This test runs with no TritonService and no framework, so looking up a replacement
  // server always fails. Check that retry() handles that failure cleanly: no exception
  // escapes, updateServer() is never called, and the action disarms itself.
  ensurePluginManager();
  edm::ParameterSet empty;
  TestTritonClient client;
  client.start();  // sets up the client's own retry bookkeeping, as the framework would

  auto action = RetryActionFactory::get()->create(
      "TritonRetryActionDifferentServer", empty, static_cast<SonicClientBase&>(client));

  action->start();
  REQUIRE(action->shouldRetry());

  REQUIRE_NOTHROW(action->retry());
  REQUIRE(client.lastServerName().empty());  // updateServer() was never reached

  REQUIRE_FALSE(action->shouldRetry());  // one-shot: disarmed after a single attempt
}

// A client whose updateServer() always fails, to exercise retry()'s error handling.
class ThrowingTritonClient : public TritonClient {
public:
  ThrowingTritonClient() : TritonClient() {}
  void updateServer(const std::string&) override { throw TritonException("updateServer failure"); }

  using TritonClient::start;

  // Bumped whenever evaluate() runs. We use this to check that a failed retry doesn't just
  // get swallowed: evaluate() only runs if finish(false) reached the client and its own
  // retry chain picked up the failure and tried again.
  int evaluateCalls = 0;

protected:
  void evaluate() override { ++evaluateCalls; }
};

TEST_CASE("TritonRetryActionDifferentServer catches exceptions from updateServer",
          "[TritonRetryActionDifferentServer]") {
  ensurePluginManager();
  edm::ParameterSet empty;
  ThrowingTritonClient client;
  client.start();
  auto action = RetryActionFactory::get()->create(
      "TritonRetryActionDifferentServer", empty, static_cast<SonicClientBase&>(client));
  action->start();

  // retry() must not let updateServer()'s exception escape...
  REQUIRE_NOTHROW(action->retry());

  // ...but swallowing it silently would be a bug: the client would never hear about the
  // failure and the call would hang forever. evaluateCalls == 1 proves the failure actually
  // reached the client (via finish(false)) and its retry chain ran.
  CHECK(client.evaluateCalls == 1);
}
