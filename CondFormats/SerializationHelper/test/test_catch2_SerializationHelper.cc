// -*- C++ -*-
//
// Package:     CondFormats/SerializationHelper
// Class  :     test_catch2_SerializationHelper
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 May 2023 19:12:55 GMT
//

// system include files
#include "catch.hpp"

// user include files
#include "CondFormats/SerializationHelper/interface/SerializationHelper.h"

namespace {
  struct Test {
    Test() = default;
    Test(float iA, int iB) : a_{iA}, b_{iB} {}

    float a_;
    int b_;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar& a_;
      ar& b_;
      ++counter_;
    }

    int counter_ = 0;
  };

  struct TestBase {
    virtual ~TestBase() noexcept {}

    virtual int value() const = 0;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {}
  };

  struct TestInheritance : public TestBase {
    ~TestInheritance() noexcept override = default;

    int value() const final { return value_; }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar& value_;
    }
    int value_ = 3145;
  };

}  // namespace

namespace cond::serialization {
  template <>
  struct ClassName<Test> {
    constexpr static std::string_view kName = "Test";
  };
  template <>
  struct ClassName<TestBase> {
    [[maybe_unused]] constexpr static std::string_view kName = "TestBase";
  };
  template <>
  struct ClassName<TestInheritance> {
    constexpr static std::string_view kName = "TestInheritance";
  };

  template <>
  struct BaseClassInfo<TestBase> : public BaseClassInfoImpl<TestBase, true, TestInheritance> {};
}  // namespace cond::serialization

using namespace cond::serialization;

TEST_CASE("Test SerializationHelper", "[SerializationHelper]") {
  SECTION("serialize") {
    SerializationHelper<Test> helper;

    std::stringbuf dataBuffer;

    Test test{1.3, -4};
    REQUIRE(test.counter_ == 0);

    auto typeName = helper.serialize(dataBuffer, &test);

    REQUIRE(dataBuffer.str().size() > 0);
    REQUIRE(test.counter_ == 1);
    REQUIRE(typeName == "Test");
  }

  SECTION("deserialize") {
    SerializationHelper<Test> helper;

    std::stringbuf dataBuffer;
    std::string_view typeName;
    {
      Test test{1.3, -4};
      REQUIRE(test.counter_ == 0);

      typeName = helper.serialize(dataBuffer, &test);
    }

    auto voidPtr = helper.deserialize(dataBuffer, typeName);

    const Test* pTest = static_cast<const Test*>(voidPtr.get());
    REQUIRE_THAT(pTest->a_, Catch::Matchers::WithinAbs(1.3, 0.001));
    REQUIRE(pTest->b_ == -4);
    REQUIRE(pTest->counter_ == 1);
  }

  SECTION("polymorphic serialize") {
    SerializationHelper<TestBase> helper;

    std::stringbuf dataBuffer;

    TestInheritance test{};

    auto typeName = helper.serialize(dataBuffer, &test);

    REQUIRE(typeName == "TestInheritance");
    REQUIRE(dataBuffer.str().size() > 0);
    REQUIRE(test.value() == 3145);
  }

  SECTION("deserialize") {
    SerializationHelper<TestBase> helper;

    std::stringbuf dataBuffer;
    std::string_view typeName;
    {
      TestInheritance test;

      typeName = helper.serialize(dataBuffer, &test);
    }

    auto voidPtr = helper.deserialize(dataBuffer, typeName);
    REQUIRE(voidPtr.get() != nullptr);

    const TestBase* pTest = static_cast<const TestBase*>(voidPtr.get());
    REQUIRE(pTest->value() == 3145);
  }
}
