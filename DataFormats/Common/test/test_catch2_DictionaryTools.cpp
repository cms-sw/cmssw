// Test of the DictionaryTools functions.

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "catch2/catch_all.hpp"
#include <typeinfo>
#include <map>
#include <vector>
#include <string>

// Helper templates for demangling checks
namespace {
  template <typename T>
  void checkIt() {
    edm::TypeWithDict type(typeid(T));
    if (bool(type)) {
      std::string demangledName(edm::typeDemangle(typeid(T).name()));
      REQUIRE(type.name() == demangledName);
      edm::TypeID tid(typeid(T));
      REQUIRE(tid.className() == demangledName);
      edm::TypeWithDict typeFromName = edm::TypeWithDict::byName(demangledName);
      REQUIRE(typeFromName.name() == demangledName);
      if (type.isClass()) {
        edm::TypeID tidFromName(typeFromName.typeInfo());
        REQUIRE(tidFromName.className() == demangledName);
      }
    }
  }
  template <typename T>
  void checkDemangling() {
    checkIt<std::vector<T> >();
    checkIt<edm::Wrapper<T> >();
    checkIt<edm::Wrapper<std::vector<T> > >();
    checkIt<T>();
    checkIt<T[1]>();
  }
}  // namespace

TEST_CASE("DictionaryTools functions", "[DictionaryTools]") {
  SECTION("default_is_invalid") {
    edm::TypeWithDict t;
    REQUIRE(!t);
  }

  SECTION("no_dictionary_is_invalid") {
    edm::TypeWithDict t(edm::TypeWithDict::byName("ThereIsNoTypeWithThisName"));
    REQUIRE(!t);
  }

  SECTION("not_a_template_instance") {
    edm::TypeWithDict not_a_template(edm::TypeWithDict::byName("double"));
    REQUIRE(not_a_template);
    std::string nonesuch(not_a_template.templateName());
    REQUIRE(nonesuch.empty());
  }

  SECTION("demangling") {
    checkDemangling<int>();
    checkDemangling<unsigned int>();
    checkDemangling<unsigned long>();
    checkDemangling<long>();
    checkDemangling<unsigned long>();
    checkDemangling<long long>();
    checkDemangling<unsigned long long>();
    checkDemangling<short>();
    checkDemangling<unsigned short>();
    checkDemangling<char>();
    checkDemangling<unsigned char>();
    checkDemangling<float>();
    checkDemangling<double>();
    checkDemangling<bool>();
    checkDemangling<std::string>();
  }
}
