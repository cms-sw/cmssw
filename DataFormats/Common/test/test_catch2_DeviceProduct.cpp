#include <atomic>
#include <memory>
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

  void resetGlobalState() {
    g_defaultConstructedCount = 0;
    g_uninitializedConstructedCount = 0;
    g_valueConstructedCount = 0;
    g_destructedCount = 0;
    g_synchronized = false;
  }
}  // namespace

// Mock Metadata class that provides synchronize() member function
class MockMetadata {
public:
  MockMetadata() = default;

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

  ~MockData() { ++g_destructedCount; }

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
