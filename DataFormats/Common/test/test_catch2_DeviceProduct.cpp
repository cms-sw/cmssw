#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <type_traits>

#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Common/interface/Uninitialized.h"

// Global atomic state for tracking object destruction
namespace {
  std::atomic<int> g_defaultConstructedCount{0};
  std::atomic<int> g_uninitializedConstructedCount{0};
  std::atomic<int> g_valueConstructedCount{0};
  std::atomic<int> g_destructedCount{0};
  std::atomic<bool> g_synchronized{false};
  std::atomic<int> g_dataDestructedBeforeMetadataCount{0};
  std::atomic<int> g_metadataDestructorCalledCount{0};

  // Track which data instances have been destructed using their unique values
  std::atomic<int> g_destructedDataValues[10] = {};  // Support up to 10 concurrent tests
  std::atomic<int> g_destructedDataCount{0};

  void resetGlobalState() {
    g_defaultConstructedCount = 0;
    g_uninitializedConstructedCount = 0;
    g_valueConstructedCount = 0;
    g_destructedCount = 0;
    g_synchronized = false;
    g_dataDestructedBeforeMetadataCount = 0;
    g_metadataDestructorCalledCount = 0;
    g_destructedDataCount = 0;
    for (auto& v : g_destructedDataValues) {
      v = 0;
    }
  }
}  // namespace

// Mock Metadata class that provides synchronize() member function
class MockMetadata : public edm::DeviceProductMetadataBase {
public:
  MockMetadata() = default;
  ~MockMetadata() noexcept override = default;

  void synchronize() const noexcept final { g_synchronized = true; }

  template <typename... Args>
  void synchronize(MockMetadata&, Args&&...) const {
    g_synchronized = true;
  }
};

// Mock data type class that tracks construction/destruction via global state
class MockData {
public:
  MockData() : value_(0) { ++g_defaultConstructedCount; }

  explicit MockData(edm::Uninitialized) : value_(-1) { ++g_uninitializedConstructedCount; }

  explicit MockData(int value) : value_(value) { ++g_valueConstructedCount; }

  MockData(const MockData&) = delete;
  MockData& operator=(const MockData&) = delete;

  MockData(MockData&& other) noexcept : value_(other.value_) { other.value_ = 0; }

  MockData& operator=(MockData&& other) noexcept {
    if (this != &other) {
      value_ = other.value_;
      other.value_ = 0;
    }
    return *this;
  }

  ~MockData() {
    // Record this data's value in the global list before incrementing the count
    int idx = g_destructedDataCount.fetch_add(1);
    if (idx < 10) {
      g_destructedDataValues[idx].store(value_);
    }
    ++g_destructedCount;
  }

  int getValue() const { return value_; }
  void setValue(int value) { value_ = value; }

private:
  int value_;
};

// Simple data type for testing without uninitialized constructor
class SimpleData {
public:
  SimpleData() : value_(42) {}
  explicit SimpleData(int value) : value_(value) {}

  SimpleData(const SimpleData&) = delete;
  SimpleData& operator=(const SimpleData&) = delete;
  SimpleData(SimpleData&&) = default;
  SimpleData& operator=(SimpleData&&) = default;

  int getValue() const { return value_; }

private:
  int value_;
};

// Metadata class that spawns a thread and demonstrates the behavior when an external thread is involved
class ThreadedMetadata : public edm::DeviceProductMetadataBase {
public:
  ThreadedMetadata(int expectedDataValue)
      : shouldStop_(false), syncCount_(0), threadReady_(false), expectedDataValue_(expectedDataValue) {
    // Start a background thread that will keep running
    workerThread_ = std::thread([this]() {
      // Signal that thread has started
      threadReady_.store(true);

      while (!shouldStop_.load()) {
        // Simulate some work that the metadata might be doing
        // Prevent overflow by capping the counter
        if (syncCount_.load() < 1000000) {
          ++syncCount_;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  ~ThreadedMetadata() noexcept override {
    synchronize();

    // Now we need to stop and join the worker thread
    shouldStop_.store(true);
    if (workerThread_.joinable()) {
      workerThread_.join();
    }
  }

  void synchronize() const noexcept override {
    if (synchronized_.load()) {
      return;  // Already synchronized
    }

    // Check if OUR specific paired data was already destructed
    int count = g_destructedDataCount.load();
    bool foundOurData = false;
    for (int i = 0; i < count; ++i) {
      if (g_destructedDataValues[i].load() == expectedDataValue_) {
        foundOurData = true;
        break;
      }
    }

    if (foundOurData) {
      ++g_dataDestructedBeforeMetadataCount;
    }

    ++g_metadataDestructorCalledCount;
    synchronized_.store(true);
  }

  int getSyncCount() const { return syncCount_.load(); }

  void waitUntilThreadReady() const {
    // Spinloop until thread is ready
    while (!threadReady_.load()) {
    }
  }

private:
  std::thread workerThread_;
  mutable std::atomic<bool> synchronized_;
  std::atomic<bool> shouldStop_;
  std::atomic<int> syncCount_;
  std::atomic<bool> threadReady_;
  int expectedDataValue_;
};

TEST_CASE("DeviceProduct default constructor", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("Default construction with MockData") {
    edm::DeviceProduct<MockData> product;
    REQUIRE(g_defaultConstructedCount == 1);
    REQUIRE(g_destructedCount == 0);
  }

  // Verify destruction
  REQUIRE(g_destructedCount == 1);
}

TEST_CASE("DeviceProduct uninitialized constructor", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("Construction with Uninitialized") {
    edm::DeviceProduct<MockData> product(edm::kUninitialized);
    REQUIRE(g_uninitializedConstructedCount == 1);
    REQUIRE(g_defaultConstructedCount == 0);
    REQUIRE(g_destructedCount == 0);
  }

  // Verify destruction
  REQUIRE(g_destructedCount == 1);
}

TEST_CASE("DeviceProduct constructor with metadata and args", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("Construction with metadata and value") {
    auto metadata = std::make_shared<MockMetadata>();
    edm::DeviceProduct<MockData> product(metadata, 100);

    REQUIRE(g_valueConstructedCount == 1);
    REQUIRE(g_defaultConstructedCount == 0);
    REQUIRE(g_uninitializedConstructedCount == 0);
  }

  REQUIRE(g_destructedCount == 1);
}

TEST_CASE("DeviceProduct metadata access", "[DeviceProduct]") {
  resetGlobalState();

  auto metadata = std::make_shared<MockMetadata>();
  edm::DeviceProduct<MockData> product(metadata, 50);

  SECTION("Metadata can be retrieved") {
    // Just verify we can call it without throwing
    REQUIRE_NOTHROW([&]() {
      auto const& md = product.metadata<MockMetadata>();
      (void)md;
    }());
  }
}

TEST_CASE("DeviceProduct getSynchronized", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("getSynchronized calls synchronize and returns data") {
    auto metadata = std::make_shared<MockMetadata>();
    edm::DeviceProduct<MockData> product(metadata, 75);

    REQUIRE(g_synchronized == false);

    MockMetadata otherMetadata;
    auto const& data = product.getSynchronized<MockMetadata>(otherMetadata);

    REQUIRE(g_synchronized == true);
    REQUIRE(data.getValue() == 75);
  }

  SECTION("getSynchronized with arguments") {
    auto metadata = std::make_shared<MockMetadata>();
    edm::DeviceProduct<MockData> product(metadata, 88);

    g_synchronized = false;

    MockMetadata otherMetadata;
    auto const& data = product.getSynchronized<MockMetadata>(otherMetadata, 1, 2, 3);

    REQUIRE(g_synchronized == true);
    REQUIRE(data.getValue() == 88);
  }
}

TEST_CASE("DeviceProduct move semantics", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("Move construction") {
    auto metadata = std::make_shared<MockMetadata>();
    edm::DeviceProduct<MockData> product1(metadata, 200);

    REQUIRE(g_valueConstructedCount == 1);

    edm::DeviceProduct<MockData> product2(std::move(product1));

    // Should not create additional MockData objects
    REQUIRE(g_valueConstructedCount == 1);

    // Verify the moved-to object works
    MockMetadata otherMetadata;
    auto const& data = product2.getSynchronized<MockMetadata>(otherMetadata);
    REQUIRE(data.getValue() == 200);
  }

  SECTION("Move assignment") {
    auto metadata1 = std::make_shared<MockMetadata>();
    auto metadata2 = std::make_shared<MockMetadata>();

    edm::DeviceProduct<MockData> product1(metadata1, 300);
    edm::DeviceProduct<MockData> product2(metadata2, 400);

    int countBefore = g_valueConstructedCount.load();

    product2 = std::move(product1);

    // Should not create additional MockData objects
    REQUIRE(g_valueConstructedCount == countBefore);
  }
}

TEST_CASE("DeviceProduct is not copyable", "[DeviceProduct]") {
  STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<edm::DeviceProduct<MockData>>);
  STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<edm::DeviceProduct<MockData>>);
}

TEST_CASE("DeviceProduct is movable", "[DeviceProduct]") {
  STATIC_REQUIRE(std::is_move_constructible_v<edm::DeviceProduct<MockData>>);
  STATIC_REQUIRE(std::is_move_assignable_v<edm::DeviceProduct<MockData>>);
}

TEST_CASE("DeviceProduct destruction tracking", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("Single object destruction") {
    {
      auto metadata = std::make_shared<MockMetadata>();
      edm::DeviceProduct<MockData> product(metadata, 123);
      REQUIRE(g_destructedCount == 0);
    }
    REQUIRE(g_destructedCount == 1);
  }

  SECTION("Multiple objects destruction") {
    {
      auto metadata1 = std::make_shared<MockMetadata>();
      auto metadata2 = std::make_shared<MockMetadata>();
      auto metadata3 = std::make_shared<MockMetadata>();

      edm::DeviceProduct<MockData> product1(metadata1, 1);
      edm::DeviceProduct<MockData> product2(metadata2, 2);
      edm::DeviceProduct<MockData> product3(metadata3, 3);

      REQUIRE(g_destructedCount == 0);
    }
    REQUIRE(g_destructedCount == 3);
  }
}

TEST_CASE("DeviceProduct race condition: data_ destructed before metadata_", "[DeviceProduct][race]") {
  resetGlobalState();

  SECTION("Ensures proper destruction order - metadata before data") {
    {
      constexpr int kSingleProductValue = 777;

      auto metadata = std::make_shared<ThreadedMetadata>(kSingleProductValue);
      edm::DeviceProduct<MockData> product(metadata, kSingleProductValue);
      // Wait for the thread to be ready and do some work
      metadata->waitUntilThreadReady();

      // Spinloop until thread has done some work
      while (metadata->getSyncCount() < 5) {
      }

      // When product goes out of scope, the proper order should be:
      // 1. DeviceProductBase::metadata_ (ThreadedMetadata) should be destructed first
      // 2. ThreadedMetadata destructor joins the thread (g_destructedCount should be 0)
      // 3. Then DeviceProduct::data_ (MockData) is destructed
      // This ensures metadata can safely synchronize while data_ is still valid
    }

    // Verify the destruction order is correct
    REQUIRE(g_destructedCount == 1);                    // MockData was destructed
    REQUIRE(g_metadataDestructorCalledCount == 1);      // ThreadedMetadata destructor was called
    REQUIRE(g_dataDestructedBeforeMetadataCount == 0);  // metadata_ should be destructed before data_
  }

  SECTION("Multiple DeviceProducts with threaded metadata") {
    {
      constexpr int kMultiProduct1Value = 100;
      constexpr int kMultiProduct2Value = 200;

      auto metadata1 = std::make_shared<ThreadedMetadata>(kMultiProduct1Value);
      auto metadata2 = std::make_shared<ThreadedMetadata>(kMultiProduct2Value);

      edm::DeviceProduct<MockData> product1(metadata1, kMultiProduct1Value);
      edm::DeviceProduct<MockData> product2(metadata2, kMultiProduct2Value);

      // Wait for both threads to be ready and do some work
      metadata1->waitUntilThreadReady();
      metadata2->waitUntilThreadReady();

      while (metadata1->getSyncCount() < 3 || metadata2->getSyncCount() < 3) {
      }

      // Both products will be destroyed with proper order:
      // metadata should be cleaned up before data for each product
    }

    // Verify the destruction order is correct
    REQUIRE(g_destructedCount == 2);
    REQUIRE(g_metadataDestructorCalledCount == 2);
    REQUIRE(g_dataDestructedBeforeMetadataCount == 0);
  }
}

TEST_CASE("DeviceProduct with SimpleData", "[DeviceProduct]") {
  SECTION("Default construction") {
    edm::DeviceProduct<SimpleData> product;
    // Just verify it compiles and constructs
    REQUIRE_NOTHROW([&]() {
      edm::DeviceProduct<SimpleData> p;
      (void)p;
    }());
  }

  SECTION("Construction with metadata and value") {
    auto metadata = std::make_shared<MockMetadata>();
    edm::DeviceProduct<SimpleData> product(metadata, 999);

    MockMetadata otherMetadata;
    auto const& data = product.getSynchronized<MockMetadata>(otherMetadata);
    REQUIRE(data.getValue() == 999);
  }
}

TEST_CASE("DeviceProduct synchronization state", "[DeviceProduct]") {
  resetGlobalState();

  SECTION("Multiple getSynchronized calls") {
    auto metadata = std::make_shared<MockMetadata>();
    edm::DeviceProduct<MockData> product(metadata, 555);

    MockMetadata otherMetadata;

    // First call
    g_synchronized = false;
    product.getSynchronized<MockMetadata>(otherMetadata);
    REQUIRE(g_synchronized == true);

    // Second call
    g_synchronized = false;
    product.getSynchronized<MockMetadata>(otherMetadata);
    REQUIRE(g_synchronized == true);
  }
}
