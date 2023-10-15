#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <string>

#include "DataFormats/L1TGlobal/interface/GlobalObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {

  bool hasNoDuplicateEnums() {
    std::vector<l1t::GlobalObject> vec;
    vec.reserve(l1t::kGlobalObjectEnumStringPairs.size());
    for (auto const& [value, name] : l1t::kGlobalObjectEnumStringPairs) {
      vec.emplace_back(value);
    }
    std::sort(vec.begin(), vec.end());
    auto const it = std::unique(vec.begin(), vec.end());
    return (it == vec.end());
  }

  bool hasNoDuplicateStrings() {
    std::vector<std::string> vec;
    vec.reserve(l1t::kGlobalObjectEnumStringPairs.size());
    for (auto const& [value, name] : l1t::kGlobalObjectEnumStringPairs) {
      vec.emplace_back(name);
    }
    std::sort(vec.begin(), vec.end());
    auto const it = std::unique(vec.begin(), vec.end());
    return (it == vec.end());
  }

  bool isValidGlobalObject(unsigned int const idx, l1t::GlobalObject const& obj, std::string const& label) {
    if (idx != obj) {
      edm::LogError("testL1TGlobalObjectCatch2")
          << "Index (" << idx << ") and Enum (" << obj << ") do not match (Label = \"" << label << "\")";
      return false;
    }

    auto const enumToString = l1t::GlobalObjectEnumToString(obj);
    if (enumToString != label) {
      edm::LogError("testL1TGlobalObjectCatch2")
          << "EnumToString (\"" << enumToString << "\") and Label (\"" << label << "\") do not match"
          << " (Index = " << idx << ", Enum = " << obj << ") !!";
      return false;
    }

    auto const stringToEnum = l1t::GlobalObjectStringToEnum(label);
    if (stringToEnum != obj) {
      edm::LogError("testL1TGlobalObjectCatch2")
          << "StringToEnum (\"" << stringToEnum << "\") and Enum (" << obj << ") do not match"
          << " (Index = " << idx << ", Label = \"" << label << "\") !!";
      return false;
    }

    return true;
  }

}  // namespace

TEST_CASE("Test l1t::GlobalObject", "[l1tGlobalObject]") {
  // verify that the enums in l1t::kGlobalObjectEnumStringPairs are all unique
  SECTION("NoDuplicateEnums") { REQUIRE(hasNoDuplicateEnums()); }

  // verify that the strings in l1t::kGlobalObjectEnumStringPairs are all unique
  SECTION("NoDuplicateStrings") { REQUIRE(hasNoDuplicateStrings()); }

  // verify correctness of index, enum and string of every GlobalObject
  SECTION("TestGlobalObjects") {
    REQUIRE(isValidGlobalObject(0, l1t::gtMu, "Mu"));                     //  0
    REQUIRE(isValidGlobalObject(1, l1t::gtMuShower, "MuShower"));         //  1
    REQUIRE(isValidGlobalObject(2, l1t::gtEG, "EG"));                     //  2
    REQUIRE(isValidGlobalObject(3, l1t::gtJet, "Jet"));                   //  3
    REQUIRE(isValidGlobalObject(4, l1t::gtTau, "Tau"));                   //  4
    REQUIRE(isValidGlobalObject(5, l1t::gtETM, "ETM"));                   //  5
    REQUIRE(isValidGlobalObject(6, l1t::gtETT, "ETT"));                   //  6
    REQUIRE(isValidGlobalObject(7, l1t::gtHTT, "HTT"));                   //  7
    REQUIRE(isValidGlobalObject(8, l1t::gtHTM, "HTM"));                   //  8
    REQUIRE(isValidGlobalObject(9, l1t::gtETMHF, "ETMHF"));               //  9
    REQUIRE(isValidGlobalObject(10, l1t::gtTowerCount, "TowerCount"));    // 10
    REQUIRE(isValidGlobalObject(11, l1t::gtMinBiasHFP0, "MinBiasHFP0"));  // 11
    REQUIRE(isValidGlobalObject(12, l1t::gtMinBiasHFM0, "MinBiasHFM0"));  // 12
    REQUIRE(isValidGlobalObject(13, l1t::gtMinBiasHFP1, "MinBiasHFP1"));  // 13
    REQUIRE(isValidGlobalObject(14, l1t::gtMinBiasHFM1, "MinBiasHFM1"));  // 14
    REQUIRE(isValidGlobalObject(15, l1t::gtETTem, "ETTem"));              // 15
    REQUIRE(isValidGlobalObject(16, l1t::gtAsymmetryEt, "AsymEt"));       // 16
    REQUIRE(isValidGlobalObject(17, l1t::gtAsymmetryHt, "AsymHt"));       // 17
    REQUIRE(isValidGlobalObject(18, l1t::gtAsymmetryEtHF, "AsymEtHF"));   // 18
    REQUIRE(isValidGlobalObject(19, l1t::gtAsymmetryHtHF, "AsymHtHF"));   // 19
    REQUIRE(isValidGlobalObject(20, l1t::gtCentrality0, "CENT0"));        // 20
    REQUIRE(isValidGlobalObject(21, l1t::gtCentrality1, "CENT1"));        // 21
    REQUIRE(isValidGlobalObject(22, l1t::gtCentrality2, "CENT2"));        // 22
    REQUIRE(isValidGlobalObject(23, l1t::gtCentrality3, "CENT3"));        // 23
    REQUIRE(isValidGlobalObject(24, l1t::gtCentrality4, "CENT4"));        // 24
    REQUIRE(isValidGlobalObject(25, l1t::gtCentrality5, "CENT5"));        // 25
    REQUIRE(isValidGlobalObject(26, l1t::gtCentrality6, "CENT6"));        // 26
    REQUIRE(isValidGlobalObject(27, l1t::gtCentrality7, "CENT7"));        // 27
    REQUIRE(isValidGlobalObject(28, l1t::gtExternal, "External"));        // 28
    REQUIRE(isValidGlobalObject(29, l1t::gtZDCP, "ZDCP"));                // 29
    REQUIRE(isValidGlobalObject(30, l1t::gtZDCM, "ZDCM"));                // 30
    REQUIRE(isValidGlobalObject(31, l1t::ObjNull, "ObjNull"));            // 31
  }
}
