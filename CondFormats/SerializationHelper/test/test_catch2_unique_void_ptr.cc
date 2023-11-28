// -*- C++ -*-
//
// Package:     CondFormats/SerializationHelper
// Class  :     test_catch2_unique_void_ptr
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 May 2023 15:24:23 GMT
//

#include "catch.hpp"
#include "CondFormats/SerializationHelper/interface/unique_void_ptr.h"

using namespace cond::serialization;

namespace {
  struct Counter {
    Counter(int& iCount) : count_(iCount) { ++count_; }
    ~Counter() { --count_; }

    int& count_;
  };

}  // namespace

TEST_CASE("Test unique_void_ptr", "[unique_void_ptr]") {
  SECTION("default constructor") {
    unique_void_ptr ptr;
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("destructor") {
    int c = 0;
    std::unique_ptr<Counter> uptr = std::make_unique<Counter>(c);
    REQUIRE(c == 1);
    {
      auto value = uptr.get();
      unique_void_ptr ptr(uptr.release(), [](const void* v) { delete static_cast<const Counter*>(v); });
      REQUIRE(value == ptr.get());
    }
    REQUIRE(c == 0);
  }

  SECTION("move constructor") {
    int c = 1;
    std::unique_ptr<Counter> uptr = std::make_unique<Counter>(c);
    REQUIRE(c == 2);
    {
      auto value = uptr.get();
      unique_void_ptr ptr(uptr.release(), [](const void* v) { delete static_cast<const Counter*>(v); });
      REQUIRE(value == ptr.get());

      {
        unique_void_ptr cpyPtr{std::move(ptr)};
        REQUIRE(ptr.get() == nullptr);
        REQUIRE(cpyPtr.get() == value);
        REQUIRE(c == 2);
      }
      REQUIRE(c == 1);
    }
    REQUIRE(c == 1);
  }

  SECTION("move operator=") {
    int c = 1;
    std::unique_ptr<Counter> uptr = std::make_unique<Counter>(c);
    REQUIRE(c == 2);
    {
      auto value = uptr.get();
      unique_void_ptr ptr(uptr.release(), [](const void* v) { delete static_cast<const Counter*>(v); });
      REQUIRE(value == ptr.get());

      {
        int c2 = 0;
        std::unique_ptr<Counter> uptr2 = std::make_unique<Counter>(c2);
        unique_void_ptr cpyPtr(uptr2.release(), [](const void* v) { delete static_cast<const Counter*>(v); });
        REQUIRE(c2 == 1);
        cpyPtr = std::move(ptr);

        REQUIRE(ptr.get() == nullptr);
        REQUIRE(cpyPtr.get() == value);
        REQUIRE(c == 2);
        REQUIRE(c2 == 0);
      }
      REQUIRE(c == 1);
    }
    REQUIRE(c == 1);
  }
}
