/**
 *
 *
 * Description: see header file.
 *
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtDefinitions.h"

// system include files
#include <cstring>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
namespace {
  template <class T>
  struct entry {
    char const* label;
    T value;
  };

  constexpr bool same(char const* x, char const* y) {
    return !*x && !*y ? true : /* default */ (*x == *y && same(x + 1, y + 1));
  }

  template <class T>
  constexpr T keyToValue(char const* label, entry<T> const* entries) {
    return !entries->label               ? entries->value
           : same(entries->label, label) ? entries->value
                                         : /*default*/ keyToValue(label, entries + 1);
  }

  template <class T>
  constexpr char const* valueToKey(T value, entry<T> const* entries) {
    return !entries->label           ? entries->label
           : entries->value == value ? entries->label
                                     : /*default*/ valueToKey(value, entries + 1);
  }
  constexpr entry<L1GtBoardType> l1GtBoardTypeStringToEnumMap[] = {{"GTFE", GTFE},
                                                                   {"FDL", FDL},
                                                                   {"PSB", PSB},
                                                                   {"GMT", GMT},
                                                                   {"TCS", TCS},
                                                                   {"TIM", TIM},
                                                                   {"BoardNull", BoardNull},
                                                                   {nullptr, (L1GtBoardType)-1}};

  constexpr entry<L1GtPsbQuad> l1GtPsbQuadStringToEnumMap[] = {{"Free", Free},
                                                               {"TechTr", TechTr},
                                                               {"IsoEGQ", IsoEGQ},
                                                               {"NoIsoEGQ", NoIsoEGQ},
                                                               {"CenJetQ", CenJetQ},
                                                               {"ForJetQ", ForJetQ},
                                                               {"TauJetQ", TauJetQ},
                                                               {"ESumsQ", ESumsQ},
                                                               {"JetCountsQ", JetCountsQ},
                                                               {"MQB1", MQB1},
                                                               {"MQB2", MQB2},
                                                               {"MQF3", MQF3},
                                                               {"MQF4", MQF4},
                                                               {"MQB5", MQB5},
                                                               {"MQB6", MQB6},
                                                               {"MQF7", MQF7},
                                                               {"MQF8", MQF8},
                                                               {"MQB9", MQB9},
                                                               {"MQB10", MQB10},
                                                               {"MQF11", MQF11},
                                                               {"MQF12", MQF12},
                                                               {"CastorQ", CastorQ},
                                                               {"HfQ", HfQ},
                                                               {"BptxQ", BptxQ},
                                                               {"GtExternalQ", GtExternalQ},
                                                               {"PsbQuadNull", PsbQuadNull},
                                                               {nullptr, (L1GtPsbQuad)-1}};

  // L1GtConditionType
  constexpr entry<L1GtConditionType> l1GtConditionTypeStringToEnumMap[] = {{"TypeNull", TypeNull},
                                                                           {"Type1s", Type1s},
                                                                           {"Type2s", Type2s},
                                                                           {"Type2wsc", Type2wsc},
                                                                           {"Type2cor", Type2cor},
                                                                           {"Type3s", Type3s},
                                                                           {"Type4s", Type4s},
                                                                           {"TypeETM", TypeETM},
                                                                           {"TypeETT", TypeETT},
                                                                           {"TypeHTT", TypeHTT},
                                                                           {"TypeHTM", TypeHTM},
                                                                           {"TypeJetCounts", TypeJetCounts},
                                                                           {"TypeCastor", TypeCastor},
                                                                           {"TypeHfBitCounts", TypeHfBitCounts},
                                                                           {"TypeHfRingEtSums", TypeHfRingEtSums},
                                                                           {"TypeBptx", TypeBptx},
                                                                           {"TypeExternal", TypeExternal},
                                                                           {nullptr, (L1GtConditionType)-1}};

  // L1GtConditionCategory
  constexpr entry<L1GtConditionCategory> l1GtConditionCategoryStringToEnumMap[] = {
      {"CondNull", CondNull},
      {"CondMuon", CondMuon},
      {"CondCalo", CondCalo},
      {"CondEnergySum", CondEnergySum},
      {"CondJetCounts", CondJetCounts},
      {"CondCorrelation", CondCorrelation},
      {"CondCastor", CondCastor},
      {"CondHfBitCounts", CondHfBitCounts},
      {"CondHfRingEtSums", CondHfRingEtSums},
      {"CondBptx", CondBptx},
      {"CondExternal", CondExternal},
      {nullptr, (L1GtConditionCategory)-1}};

}  // namespace
// L1GtBoardType
L1GtBoardType l1GtBoardTypeStringToEnum(const std::string& label) {
  L1GtBoardType value = keyToValue(label.c_str(), l1GtBoardTypeStringToEnumMap);
  if (value == (L1GtBoardType)-1) {
    edm::LogInfo("L1GtDefinitions") << "\n  '" << label << "' is not a recognized L1GtBoardType. \n  Return BoardNull.";
    value = BoardNull;
  }

  if (value == BoardNull) {
    edm::LogInfo("L1GtDefinitions") << "\n  BoardNull means no valid board type defined!";
  }

  return value;
}

std::string l1GtBoardTypeEnumToString(const L1GtBoardType& boardType) {
  char const* result = valueToKey(boardType, l1GtBoardTypeStringToEnumMap);
  if (boardType == BoardNull) {
    edm::LogInfo("L1GtDefinitions") << "\n  BoardNull means no valid board type defined!";
  }
  if (!result) {
    edm::LogInfo("L1GtDefinitions") << "\n  '" << boardType << "' is not a recognized L1GtBoardType. "
                                    << "\n  Return BoardNull, which means no valid board type defined!";
    return "BoardNull";
  }
  return result;
}

// L1GtPsbQuad

L1GtPsbQuad l1GtPsbQuadStringToEnum(const std::string& label) {
  L1GtPsbQuad value = keyToValue(label.c_str(), l1GtPsbQuadStringToEnumMap);
  // in case of unrecognized L1GtPsbQuad, return PsbQuadNull
  // to be dealt by the corresponding module
  if (value == (L1GtPsbQuad)-1) {
    edm::LogInfo("L1GtDefinitions") << "\n  '" << label << "' is not a recognized L1GtPsbQuad. \n  Return PsbQuadNull.";
    value = PsbQuadNull;
  }

  if (value == PsbQuadNull) {
    edm::LogInfo("L1GtDefinitions") << "\n  PsbQuadNull means no valid PSB quadruplet defined!";
  }

  return value;
}

std::string l1GtPsbQuadEnumToString(const L1GtPsbQuad& psbQuad) {
  char const* result = valueToKey(psbQuad, l1GtPsbQuadStringToEnumMap);
  if (psbQuad == PsbQuadNull)
    edm::LogInfo("L1GtDefinitions") << "\n  PsbQuadNull means no valid PSB quadruplet defined!";
  if (!result) {
    result = "PsbQuadNull";
    edm::LogInfo("L1GtDefinitions") << "\n  '" << psbQuad << "' is not a recognized L1GtPsbQuad. "
                                    << "\n  Return PsbQuadNull, which means no valid PSB quadruplet defined!";
  }

  return result;
}

L1GtConditionType l1GtConditionTypeStringToEnum(const std::string& label) {
  L1GtConditionType value = keyToValue(label.c_str(), l1GtConditionTypeStringToEnumMap);

  // in case of unrecognized L1GtConditionType, return TypeNull
  // to be dealt by the corresponding module
  if (value == (L1GtConditionType)-1) {
    edm::LogInfo("L1GtDefinitions") << "\n  '" << label
                                    << "' is not a recognized L1GtConditionType. \n  Return TypeNull.";

    value = TypeNull;
  }

  if (value == TypeNull) {
    edm::LogInfo("L1GtDefinitions") << "\n  TypeNull means no valid condition type defined!";
  }

  return value;
}

std::string l1GtConditionTypeEnumToString(const L1GtConditionType& conditionType) {
  const char* result = valueToKey(conditionType, l1GtConditionTypeStringToEnumMap);
  if (conditionType == TypeNull)
    edm::LogInfo("L1GtDefinitions") << "\n  Return TypeNull, which means no valid condition type defined!";
  if (!result) {
    result = "TypeNull";
    edm::LogInfo("L1GtDefinitions") << "\n  '" << conditionType << "' is not a recognized L1GtConditionType. "
                                    << "\n  Return TypeNull, which means no valid condition type defined!";
  }
  return result;
}

L1GtConditionCategory l1GtConditionCategoryStringToEnum(const std::string& label) {
  L1GtConditionCategory value = keyToValue(label.c_str(), l1GtConditionCategoryStringToEnumMap);
  // in case of unrecognized L1GtConditionCategory, return CondNull
  // to be dealt by the corresponding module
  if (value == (L1GtConditionCategory)-1) {
    edm::LogInfo("L1GtDefinitions") << "\n  '" << label
                                    << "' is not a recognized L1GtConditionCategory. \n  Return CondNull.";

    value = CondNull;
  }

  if (value == CondNull) {
    edm::LogInfo("L1GtDefinitions") << "\n  CondNull means no valid condition category defined!";
  }

  return value;
}

std::string l1GtConditionCategoryEnumToString(const L1GtConditionCategory& conditionCategory) {
  char const* result = valueToKey(conditionCategory, l1GtConditionCategoryStringToEnumMap);
  if (conditionCategory == CondNull)
    edm::LogInfo("L1GtDefinitions") << "\n  Return CondNull, which means no valid condition category defined!";

  if (!result) {
    result = "CondNull";
    edm::LogInfo("L1GtDefinitions") << "\n  '" << conditionCategory << "' is not a recognized L1GtConditionCategory. "
                                    << "\n  Return CondNull, which means no valid condition category defined!";
  }

  return result;
}
