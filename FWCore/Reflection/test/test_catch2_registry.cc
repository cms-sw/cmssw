#include "FWCore/Reflection/interface/DataProductReflectionInfoRegistry.h"
#include <vector>

#include "catch.hpp"

namespace test {
  struct Foo {
    virtual ~Foo() = default;
  };

  struct Bar : public Foo {};
  struct DiveBar : private Foo {};
  struct GoodBar : public Bar {};

  struct NotRegistered {};
}  // namespace test

using namespace test;

//Below must be specified by developer
DEFINE_DATA_PRODUCT_INFO(Foo);
DEFINE_DATA_PRODUCT_INFO(Bar, Foo);
DEFINE_DATA_PRODUCT_INFO(GoodBar, Bar, Foo);
DEFINE_DATA_PRODUCT_INFO(DiveBar);
DEFINE_DATA_PRODUCT_INFO(std::vector<Foo>);
//will fail to compile which is good
//DEFINE_DATA_PRODUCT_INFO(DiveBar, Foo);

TEST_CASE("Test DataProductReflectionInfoRegistry", "[DataProductReflectionInfoRegistry]") {
  SECTION("Exists") {
    REQUIRE(edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(Foo))));
    REQUIRE(edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(Bar))));
    REQUIRE(edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(DiveBar))));

    REQUIRE(edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(GoodBar))));
    REQUIRE(edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(std::vector<Foo>))));
  }
  SECTION("Missing") {
    REQUIRE(not edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(NotRegistered))));
  }
  SECTION("Check info") {
    SECTION("Base no inheritance") {
      auto fooInfo = edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(Foo)));
      REQUIRE(fooInfo);
      REQUIRE(fooInfo->typeInfo() == typeid(Foo));
      REQUIRE(not fooInfo->isContainer());
      auto range = fooInfo->inheritsFrom();
      REQUIRE(range.empty());
    }
    SECTION("Single public inheritance") {
      auto info = edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(Bar)));
      REQUIRE(info);
      REQUIRE(info->typeInfo() == typeid(Bar));
      REQUIRE(not info->isContainer());
      auto range = info->inheritsFrom();
      REQUIRE(not range.empty());
      REQUIRE(range.size() == 1);
      REQUIRE(*(*range.begin()) == typeid(Foo));
    }
    SECTION("Multiple public inheritance") {
      auto info = edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(GoodBar)));
      REQUIRE(info);
      REQUIRE(info->typeInfo() == typeid(GoodBar));
      REQUIRE(not info->isContainer());
      auto range = info->inheritsFrom();
      REQUIRE(not range.empty());
      REQUIRE(range.size() == 2);
      REQUIRE(*(*range.begin()) == typeid(Bar));
      REQUIRE(*(*(range.begin() + 1)) == typeid(Foo));
    }
    SECTION("Container, no inheritance") {
      auto info =
          edm::DataProductReflectionInfoRegistry::instance().findType(std::type_index(typeid(std::vector<Foo>)));
      REQUIRE(info);
      REQUIRE(info->typeInfo() == typeid(std::vector<Foo>));
      REQUIRE(info->isContainer());
      REQUIRE(info->elementType() == typeid(Foo));
      auto range = info->inheritsFrom();
      REQUIRE(range.empty());
    }
  }
}
