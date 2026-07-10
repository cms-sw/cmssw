// Test of the DictionaryTools functions.

#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include <catch2/catch_all.hpp>

#include <typeinfo>
#include <vector>

namespace {
  template <typename T>
  void checkIt() {
    edm::TypeWithDict type(typeid(T));
    // Test only if class has dictionary
    if (bool(type)) {
      std::string demangledName(edm::typeDemangle(typeid(T).name()));
      REQUIRE(type.name() == demangledName);
    }
  }

  template <typename T>
  void checkDemangling() {
    checkIt<T>();
    checkIt<std::vector<T> >();
  }
}  // namespace

TEST_CASE("Dictionaries", "[Enum_t]") {
  SECTION("enum_is_valid") {
    edm::TypeWithDict t(typeid(edmtest::EnumProduct::TheEnumProduct));
    REQUIRE(t);
  }

  SECTION("enum_by_name_is_valid") {
    edm::TypeWithDict t = edm::TypeWithDict::byName("edmtest::EnumProduct::TheEnumProduct");
    REQUIRE(t);
  }

  SECTION("enum_member_is_valid") {
    edm::TypeWithDict t = edm::TypeWithDict::byName("edmtest::EnumProduct");
    edm::MemberWithDict m = t.dataMemberByName("value");
    edm::TypeWithDict t2 = m.typeOf();
    edm::TypeWithDict t3 = edm::TypeWithDict::byName("edmtest::EnumProduct::TheEnumProduct");
    REQUIRE(t2);
    REQUIRE(t3);
    REQUIRE(t2 == t3);
  }

  SECTION("array_member_is_valid") {
    edm::TypeWithDict t = edm::TypeWithDict::byName("edmtest::ArrayProduct");
    edm::MemberWithDict m = t.dataMemberByName("value");
    REQUIRE(m.isArray());
    edm::TypeWithDict t2 = m.typeOf();
    edm::TypeWithDict t3 = edm::TypeWithDict::byName("int[1]");
    REQUIRE(t2);
    REQUIRE(t3);
    REQUIRE(t2.qualifiedName() == "int[1]");
    REQUIRE(t2 == t3);
  }

  SECTION("demangling") { checkDemangling<edmtest::EnumProduct::TheEnumProduct>(); }
}
