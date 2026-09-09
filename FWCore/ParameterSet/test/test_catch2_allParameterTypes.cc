// Checks that every supported parameter type maps to its enum, is written
// correctly into an untyped cfi, and survives as part of a VPSet default.

#include "catch2/catch_all.hpp"

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cctype>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

  template <typename T>
  constexpr bool isVectorParameter = false;
  template <typename T>
  constexpr bool isVectorParameter<std::vector<T>> = true;

  // Applies f to a sample value of every supported parameter type except PSet
  // and VPSet. std::vector<bool> is not supported as a parameter type.
  template <typename F>
  void forEachParameterType(F f) {
    f(1);
    f(std::vector<int>({1, 2}));
    f(1u);
    f(std::vector<unsigned>({1u, 2u}));
    f(1LL);
    f(std::vector<long long>({1LL, 2LL}));
    f(1ULL);
    f(std::vector<unsigned long long>({1ULL, 2ULL}));
    f(1.5);
    f(std::vector<double>({1.5, 2.5}));
    f(1.5f);
    f(std::vector<float>({1.5f, 2.5f}));
    f(true);
    f(std::string("a"));
    f(std::vector<std::string>({std::string("a"), std::string("b")}));
    f(edm::EventID(1, 1, 1));
    f(std::vector<edm::EventID>({edm::EventID(1, 1, 1)}));
    f(edm::LuminosityBlockID(1, 1));
    f(std::vector<edm::LuminosityBlockID>({edm::LuminosityBlockID(1, 1)}));
    f(edm::InputTag("a"));
    f(std::vector<edm::InputTag>({edm::InputTag("a")}));
    f(edm::ESInputTag("", "a"));
    f(std::vector<edm::ESInputTag>({edm::ESInputTag("", "a")}));
    f(edm::LuminosityBlockRange(1, 1, 2, 2));
    f(std::vector<edm::LuminosityBlockRange>({edm::LuminosityBlockRange(1, 1, 2, 2)}));
    f(edm::EventRange(1, 1, 1, 2, 2, 2));
    f(std::vector<edm::EventRange>({edm::EventRange(1, 1, 1, 2, 2, 2)}));
    f(edm::FileInPath());
  }

  template <typename T>
  std::string parameterName() {
    return "a" + edm::parameterTypeEnumToString(edm::ParameterTypeToEnum::toEnum<T>());
  }

  std::string writeUntypedCfi(edm::ParameterDescriptionNode const& node) {
    std::ostringstream os;
    edm::CfiOptions options = edm::cfi::Untyped{edm::cfi::Paths{}};
    bool startWithComma = false;
    bool wroteSomething = false;
    node.writeCfi(os, edm::ParameterModifier::kNone, startWithComma, 0, options, wroteSomething);
    return os.str();
  }

  // A vector parameter must be written as a bracketed list, a scalar must not.
  void checkBrackets(std::string const& written, bool isVector) {
    auto const pos = written.find('=');
    REQUIRE(pos != std::string::npos);
    std::string stripped;
    for (char c : written.substr(pos + 1)) {
      if (not std::isspace(static_cast<unsigned char>(c))) {
        stripped += c;
      }
    }
    INFO("written cfi: " << written);
    REQUIRE(not stripped.empty());
    if (isVector) {
      CHECK(stripped.front() == '[');
      CHECK(stripped.back() == ']');
    } else {
      CHECK(stripped.front() != '[');
    }
  }
}  // namespace

TEST_CASE("every parameter type is handled consistently", "[ParameterTypes]") {
  SECTION("type maps to its enum and writes a well formed untyped cfi") {
    int count = 0;
    forEachParameterType([&count](auto const& value) {
      using T = std::decay_t<decltype(value)>;
      ++count;
      edm::ParameterDescription<T> description(parameterName<T>(), value, true);
      CHECK(description.type() == edm::ParameterTypeToEnum::toEnum<T>());
      checkBrackets(writeUntypedCfi(description), isVectorParameter<T>);
    });
    CHECK(count == 28);

    edm::ParameterSetDescription nested;
    nested.add<float>("aFloat", 1.5f);
    edm::ParameterDescription<edm::ParameterSetDescription> pset("aPSet", nested, true);
    CHECK(pset.type() == edm::k_PSet);
    edm::ParameterDescription<std::vector<edm::ParameterSet>> vpset(
        "aVPSet", nested, true, std::vector<edm::ParameterSet>());
    CHECK(vpset.type() == edm::k_VPSet);
  }

  SECTION("a VPSet default element keeps every parameter type in the written cfi") {
    edm::ParameterSet element;
    std::vector<std::string> names;
    forEachParameterType([&element, &names](auto const& value) {
      using T = std::decay_t<decltype(value)>;
      names.push_back(parameterName<T>());
      element.addParameter<T>(names.back(), value);
    });
    REQUIRE(names.size() == 28);

    edm::ParameterSet nested;
    nested.addParameter<float>("aNestedFloat", 2.5f);
    element.addParameter<edm::ParameterSet>("aPSet", nested);
    element.addParameter<std::vector<edm::ParameterSet>>("aVPSet", std::vector<edm::ParameterSet>(1, nested));
    names.insert(names.end(), {"aPSet", "aVPSet", "aNestedFloat"});

    edm::ParameterSetDescription elementDescription;
    edm::ParameterDescription<std::vector<edm::ParameterSet>> vpset(
        "theVPSet", elementDescription, true, std::vector<edm::ParameterSet>(1, element));
    std::ostringstream os;
    edm::CfiOptions options = edm::cfi::Typed{};
    bool startWithComma = false;
    bool wroteSomething = false;
    vpset.writeCfi(os, edm::ParameterModifier::kNone, startWithComma, 0, options, wroteSomething);
    std::string const written = os.str();

    for (auto const& name : names) {
      INFO("written cfi: " << written);
      CHECK(written.find(name) != std::string::npos);
    }
  }
}
