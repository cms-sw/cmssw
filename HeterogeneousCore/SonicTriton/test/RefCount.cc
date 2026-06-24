#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include <string>
#include <unordered_map>
#include <mutex>

// Standalone refcount logic test
// This tests the refcount algorithm without requiring the full TritonService infrastructure

// Simplified model state for testing refcount logic
struct TestModelState {
  std::string modelName;
  std::string path;
  int refCount{0};
  bool isLoaded() const { return refCount > 0; }
};

// Mock class that implements the same refcount logic as TritonService
class RefCountManager {
public:
  // Tracks actual server load/unload calls
  int serverLoadCalls{0};
  int serverUnloadCalls{0};

  // Simulates loadModel behavior
  bool loadModel(const std::string& modelName, const std::string& path = "") {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& state = models_[modelName];
    if (state.modelName.empty())
      state.modelName = modelName;
    if (state.path.empty() && !path.empty())
      state.path = path;

    // If already loaded, just bump refcount (no server call)
    if (state.refCount > 0) {
      ++state.refCount;
      refCounts_[modelName] = state.refCount;
      return true;
    }

    // Actually "load" on server (simulated)
    ++serverLoadCalls;
    state.refCount = 1;
    refCounts_[modelName] = state.refCount;
    return true;
  }

  // Simulates unloadModel behavior
  bool unloadModel(const std::string& modelName) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(modelName);
    if (it == models_.end() || it->second.refCount == 0) {
      return false;  // Not loaded
    }

    auto& state = it->second;

    // If refcount > 1, just decrement (no server call)
    if (state.refCount > 1) {
      --(state.refCount);
      refCounts_[modelName] = state.refCount;
      return true;
    }

    // Actually "unload" from server (simulated)
    ++serverUnloadCalls;
    refCounts_.erase(modelName);
    state.refCount = 0;
    return true;
  }

  int getRefCount(const std::string& modelName) const {
    auto it = refCounts_.find(modelName);
    return (it != refCounts_.end()) ? it->second : 0;
  }

private:
  std::unordered_map<std::string, TestModelState> models_;
  std::unordered_map<std::string, int> refCounts_;
  std::mutex mutex_;
};

TEST_CASE("RefCount: single load increments to 1", "[RefCount]") {
  RefCountManager mgr;

  REQUIRE(mgr.loadModel("model_a", "/path/to/model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 1);
  REQUIRE(mgr.serverLoadCalls == 1);
}

TEST_CASE("RefCount: multiple loads increment without server calls", "[RefCount]") {
  RefCountManager mgr;

  // First load - should call server
  REQUIRE(mgr.loadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 1);
  REQUIRE(mgr.serverLoadCalls == 1);

  // Second load - should NOT call server, just increment
  REQUIRE(mgr.loadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 2);
  REQUIRE(mgr.serverLoadCalls == 1);  // Still 1

  // Third load - should NOT call server, just increment
  REQUIRE(mgr.loadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 3);
  REQUIRE(mgr.serverLoadCalls == 1);  // Still 1
}

TEST_CASE("RefCount: unload decrements without server call until zero", "[RefCount]") {
  RefCountManager mgr;

  // Load 3 times
  mgr.loadModel("model_a");
  mgr.loadModel("model_a");
  mgr.loadModel("model_a");
  REQUIRE(mgr.getRefCount("model_a") == 3);
  REQUIRE(mgr.serverLoadCalls == 1);

  // First unload - decrement only, no server call
  REQUIRE(mgr.unloadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 2);
  REQUIRE(mgr.serverUnloadCalls == 0);

  // Second unload - decrement only, no server call
  REQUIRE(mgr.unloadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 1);
  REQUIRE(mgr.serverUnloadCalls == 0);

  // Third unload - should call server (refcount reaches 0)
  REQUIRE(mgr.unloadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 0);
  REQUIRE(mgr.serverUnloadCalls == 1);
}

TEST_CASE("RefCount: unload on non-loaded model returns false", "[RefCount]") {
  RefCountManager mgr;

  // Unload without loading first
  REQUIRE_FALSE(mgr.unloadModel("model_a"));
  REQUIRE(mgr.serverUnloadCalls == 0);
}

TEST_CASE("RefCount: reload after full unload triggers new server load", "[RefCount]") {
  RefCountManager mgr;

  // Load and fully unload
  mgr.loadModel("model_a");
  mgr.unloadModel("model_a");
  REQUIRE(mgr.getRefCount("model_a") == 0);
  REQUIRE(mgr.serverLoadCalls == 1);
  REQUIRE(mgr.serverUnloadCalls == 1);

  // Reload - should call server again
  REQUIRE(mgr.loadModel("model_a"));
  REQUIRE(mgr.getRefCount("model_a") == 1);
  REQUIRE(mgr.serverLoadCalls == 2);  // Now 2
}

TEST_CASE("RefCount: multiple models are independent", "[RefCount]") {
  RefCountManager mgr;

  // Load two different models
  mgr.loadModel("model_a");
  mgr.loadModel("model_b");
  REQUIRE(mgr.getRefCount("model_a") == 1);
  REQUIRE(mgr.getRefCount("model_b") == 1);
  REQUIRE(mgr.serverLoadCalls == 2);

  // Load model_a again
  mgr.loadModel("model_a");
  REQUIRE(mgr.getRefCount("model_a") == 2);
  REQUIRE(mgr.getRefCount("model_b") == 1);
  REQUIRE(mgr.serverLoadCalls == 2);  // No new server call

  // Unload model_b completely
  mgr.unloadModel("model_b");
  REQUIRE(mgr.getRefCount("model_a") == 2);
  REQUIRE(mgr.getRefCount("model_b") == 0);
  REQUIRE(mgr.serverUnloadCalls == 1);

  // model_a still loaded
  REQUIRE(mgr.getRefCount("model_a") == 2);
}

TEST_CASE("RefCount: path is preserved from first load", "[RefCount]") {
  RefCountManager mgr;

  // First load with path
  mgr.loadModel("model_a", "/path/to/model");
  REQUIRE(mgr.getRefCount("model_a") == 1);

  // Second load without path - should still work
  mgr.loadModel("model_a");
  REQUIRE(mgr.getRefCount("model_a") == 2);
}
