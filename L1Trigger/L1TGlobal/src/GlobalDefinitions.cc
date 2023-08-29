/**
 *
 *
 * Description: see header file.
 *
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *          Vladimir Rekovic - extend for overlap removal
 *          Elisa Fontanesi - extended for three-body correlation conditions
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/GlobalDefinitions.h"

// system include files
#include <cstring>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
namespace {
  template <class T>
  struct entry {
    char const *label;
    T value;
  };

  constexpr bool same(char const *x, char const *y) {
    return !*x && !*y ? true : /* default */ (*x == *y && same(x + 1, y + 1));
  }

  template <class T>
  constexpr T keyToValue(char const *label, entry<T> const *entries) {
    return !entries->label               ? entries->value
           : same(entries->label, label) ? entries->value
                                         : /*default*/ keyToValue(label, entries + 1);
  }

  template <class T>
  constexpr char const *valueToKey(T value, entry<T> const *entries) {
    return !entries->label           ? entries->label
           : entries->value == value ? entries->label
                                     : /*default*/ valueToKey(value, entries + 1);
  }
  constexpr entry<l1t::L1GtBoardType> l1GtBoardTypeStringToEnumMap[] = {
      {"l1t::MP7", l1t::MP7}, {"l1t::BoardNull", l1t::BoardNull}, {nullptr, (l1t::L1GtBoardType)-1}};

  // l1t::GtConditionType
  constexpr entry<l1t::GtConditionType> l1GtConditionTypeStringToEnumMap[] = {
      {"l1t::TypeNull", l1t::TypeNull},
      {"l1t::Type1s", l1t::Type1s},
      {"l1t::Type2s", l1t::Type2s},
      {"l1t::Type2wsc", l1t::Type2wsc},
      {"l1t::Type2cor", l1t::Type2cor},
      {"l1t::Type3s", l1t::Type3s},
      {"l1t::Type4s", l1t::Type4s},
      {"l1t::TypeETM", l1t::TypeETM},
      {"l1t::TypeETT", l1t::TypeETT},
      {"l1t::TypeHTT", l1t::TypeHTT},
      {"l1t::TypeHTM", l1t::TypeHTM},
      {"l1t::TypeETMHF", l1t::TypeETMHF},
      {"l1t::TypeTowerCount", l1t::TypeTowerCount},
      {"l1t::TypeMinBiasHFP0", l1t::TypeMinBiasHFP0},
      {"l1t::TypeMinBiasHFM0", l1t::TypeMinBiasHFM0},
      {"l1t::TypeMinBiasHFP1", l1t::TypeMinBiasHFP1},
      {"l1t::TypeMinBiasHFM1", l1t::TypeMinBiasHFM1},
      {"l1t::TypeZDCP", l1t::TypeZDCP},
      {"l1t::TypeZDCM", l1t::TypeZDCM},
      {"l1t::TypeExternal", l1t::TypeExternal},
      {nullptr, (l1t::GtConditionType)-1},
      {"l1t::Type2corWithOverlapRemoval", l1t::Type2corWithOverlapRemoval},
      {"l1t::TypeCent0", l1t::TypeCent0},
      {"l1t::TypeCent1", l1t::TypeCent1},
      {"l1t::TypeCent2", l1t::TypeCent2},
      {"l1t::TypeCent3", l1t::TypeCent3},
      {"l1t::TypeCent4", l1t::TypeCent4},
      {"l1t::TypeCent5", l1t::TypeCent5},
      {"l1t::TypeCent6", l1t::TypeCent6},
      {"l1t::TypeCent7", l1t::TypeCent7},
      {"l1t::TypeAsymEt", l1t::TypeAsymEt},
      {"l1t::TypeAsymHt", l1t::TypeAsymHt},
      {"l1t::TypeAsymEtHF", l1t::TypeAsymEtHF},
      {"l1t::TypeAsymHtHF", l1t::TypeAsymHtHF}};

  // l1t::GtConditionCategory
  constexpr entry<l1t::GtConditionCategory> l1GtConditionCategoryStringToEnumMap[] = {
      {"l1t::CondNull", l1t::CondNull},
      {"l1t::CondMuon", l1t::CondMuon},
      {"l1t::CondCalo", l1t::CondCalo},
      {"l1t::CondEnergySum", l1t::CondEnergySum},
      {"l1t::CondZdcEnergySum", l1t::CondZdcEnergySum},
      {"l1t::CondCorrelation", l1t::CondCorrelation},
      {"l1t::CondCorrelationThreeBody", l1t::CondCorrelationThreeBody},
      {"l1t::CondCorrelationWithOverlapRemoval", l1t::CondCorrelationWithOverlapRemoval},
      {"l1t::CondExternal", l1t::CondExternal},
      {nullptr, (l1t::GtConditionCategory)-1}};

}  // namespace
// l1t::L1GtBoardType
l1t::L1GtBoardType l1t::l1GtBoardTypeStringToEnum(const std::string &label) {
  l1t::L1GtBoardType value = keyToValue(label.c_str(), l1GtBoardTypeStringToEnumMap);
  if (value == (l1t::L1GtBoardType)-1) {
    edm::LogInfo("L1TGlobal") << "\n  '" << label
                              << "' is not a recognized l1t::L1GtBoardType. \n  Return l1t::BoardNull.";
    value = l1t::BoardNull;
  }

  if (value == l1t::BoardNull) {
    edm::LogInfo("L1TGlobal") << "\n  l1t::BoardNull means no valid board type defined!";
  }

  return value;
}

std::string l1t::l1GtBoardTypeEnumToString(const l1t::L1GtBoardType &boardType) {
  char const *result = valueToKey(boardType, l1GtBoardTypeStringToEnumMap);
  if (boardType == l1t::BoardNull) {
    edm::LogInfo("L1TGlobal") << "\n  l1t::BoardNull means no valid board type defined!";
  }
  if (!result) {
    edm::LogInfo("L1TGlobal") << "\n  '" << boardType << "' is not a recognized l1t::L1GtBoardType. "
                              << "\n  Return l1t::BoardNull, which means no valid board type defined!";
    return "l1t::BoardNull";
  }
  return result;
}

l1t::GtConditionType l1t::l1GtConditionTypeStringToEnum(const std::string &label) {
  l1t::GtConditionType value = keyToValue(label.c_str(), l1GtConditionTypeStringToEnumMap);

  // in case of unrecognized l1t::GtConditionType, return l1t::TypeNull
  // to be dealt by the corresponding module
  if (value == (l1t::GtConditionType)-1) {
    edm::LogInfo("L1TGlobal") << "\n  '" << label
                              << "' is not a recognized l1t::GtConditionType. \n  Return l1t::TypeNull.";

    value = l1t::TypeNull;
  }

  if (value == l1t::TypeNull) {
    edm::LogInfo("L1TGlobal") << "\n  l1t::TypeNull means no valid condition type defined!";
  }

  return value;
}

std::string l1t::l1GtConditionTypeEnumToString(const l1t::GtConditionType &conditionType) {
  const char *result = valueToKey(conditionType, l1GtConditionTypeStringToEnumMap);
  if (conditionType == l1t::TypeNull)
    edm::LogInfo("L1TGlobal") << "\n  Return l1t::TypeNull, which means no valid condition type defined!";
  if (!result) {
    result = "l1t::TypeNull";
    edm::LogInfo("L1TGlobal") << "\n  '" << conditionType << "' is not a recognized l1t::GtConditionType. "
                              << "\n  Return l1t::TypeNull, which means no valid condition type defined!";
  }
  return result;
}

l1t::GtConditionCategory l1t::l1GtConditionCategoryStringToEnum(const std::string &label) {
  l1t::GtConditionCategory value = keyToValue(label.c_str(), l1GtConditionCategoryStringToEnumMap);
  // in case of unrecognized l1t::GtConditionCategory, return l1t::CondNull
  // to be dealt by the corresponding module
  if (value == (l1t::GtConditionCategory)-1) {
    edm::LogInfo("L1TGlobal") << "\n  '" << label
                              << "' is not a recognized l1t::GtConditionCategory. \n  Return l1t::CondNull.";

    value = l1t::CondNull;
  }

  if (value == l1t::CondNull) {
    edm::LogInfo("L1TGlobal") << "\n  l1t::CondNull means no valid condition category defined!";
  }

  return value;
}

std::string l1t::l1GtConditionCategoryEnumToString(const l1t::GtConditionCategory &conditionCategory) {
  char const *result = valueToKey(conditionCategory, l1GtConditionCategoryStringToEnumMap);
  if (conditionCategory == l1t::CondNull)
    edm::LogInfo("L1TGlobal") << "\n  Return l1t::CondNull, which means no valid condition category defined!";

  if (!result) {
    result = "l1t::CondNull";
    edm::LogInfo("L1TGlobal") << "\n  '" << conditionCategory << "' is not a recognized l1t::GtConditionCategory. "
                              << "\n  Return l1t::CondNull, which means no valid condition category defined!";
  }

  return result;
}
