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
#include "L1Trigger/L1TGlobal/interface/GtDefinitions.h"

// system include files
#include <cstring>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
namespace {
template <class T>
struct entry {                                                                                                                                                    
  char const* label;
  T           value;
};

constexpr bool same(char const *x, char const *y) {
  return !*x && !*y     ? true                                                                                                                                
       : /* default */    (*x == *y && same(x+1, y+1));                                                                                                       
}

template <class T>
constexpr T keyToValue(char const *label, entry<T> const *entries) {                                                                                  
  return !entries->label ? entries->value                         
       : same(entries->label, label) ? entries->value
       : /*default*/                   keyToValue(label, entries+1);                                                                                 
}

template <class T>
constexpr char const*valueToKey(T value, entry<T> const *entries) {
  return !entries->label ? entries->label
       : entries->value == value ? entries->label
       : /*default*/       valueToKey(value, entries+1);
}
constexpr entry<l1t::L1GtBoardType> l1GtBoardTypeStringToEnumMap[] = {
            {"l1t::GTFE", l1t::GTFE},
            {"l1t::FDL", l1t::FDL},
            {"l1t::PSB", l1t::PSB},
            {"l1t::GMT", l1t::GMT},
            {"l1t::TCS", l1t::TCS},
            {"l1t::TIM", l1t::TIM},
            {"l1t::BoardNull", l1t::BoardNull},
            {0, (l1t::L1GtBoardType)-1}
};

constexpr entry<l1t::L1GtPsbQuad> l1GtPsbQuadStringToEnumMap[] = {
        {"l1t::Free", l1t::Free},
        {"l1t::TechTr", l1t::TechTr},
        {"l1t::IsoEGQ", l1t::IsoEGQ},
        {"l1t::NoIsoEGQ", l1t::NoIsoEGQ},
        {"l1t::CenJetQ", l1t::CenJetQ},
        {"l1t::ForJetQ", l1t::ForJetQ},
        {"l1t::TauJetQ", l1t::TauJetQ},
        {"l1t::ESumsQ", l1t::ESumsQ},
        {"l1t::JetCountsQ", l1t::JetCountsQ},
        {"l1t::MQB1", l1t::MQB1},
        {"l1t::MQB2", l1t::MQB2},
        {"l1t::MQF3", l1t::MQF3},
        {"l1t::MQF4", l1t::MQF4},
        {"l1t::MQB5", l1t::MQB5},
        {"l1t::MQB6", l1t::MQB6},
        {"l1t::MQF7", l1t::MQF7},
        {"l1t::MQF8", l1t::MQF8},
        {"l1t::MQB9", l1t::MQB9},
        {"l1t::MQB10", l1t::MQB10},
        {"l1t::MQF11", l1t::MQF11},
        {"l1t::MQF12", l1t::MQF12},
        {"l1t::CastorQ", l1t::CastorQ},
        {"l1t::HfQ", l1t::HfQ},
        {"l1t::BptxQ", l1t::BptxQ},
        {"l1t::GtExternalQ", l1t::GtExternalQ},
        {"l1t::PsbQuadNull", l1t::PsbQuadNull},
        {0, (l1t::L1GtPsbQuad) - 1}
};

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
        {"l1t::TypeJetCounts", l1t::TypeJetCounts},
        {"l1t::TypeCastor", l1t::TypeCastor},
        {"l1t::TypeHfBitCounts", l1t::TypeHfBitCounts},
        {"l1t::TypeHfRingEtSums", l1t::TypeHfRingEtSums},
        {"l1t::TypeBptx", l1t::TypeBptx},
        {"l1t::TypeExternal", l1t::TypeExternal},
        {0, (l1t::GtConditionType) - 1}
};

// l1t::GtConditionCategory
constexpr entry<l1t::GtConditionCategory> l1GtConditionCategoryStringToEnumMap[] = {
  {"l1t::CondNull", l1t::CondNull},
  {"l1t::CondMuon", l1t::CondMuon},
  {"l1t::CondCalo", l1t::CondCalo},
  {"l1t::CondEnergySum", l1t::CondEnergySum},
  {"l1t::CondJetCounts", l1t::CondJetCounts},
  {"l1t::CondCorrelation", l1t::CondCorrelation},
  {"l1t::CondCastor", l1t::CondCastor},
  {"l1t::CondHfBitCounts", l1t::CondHfBitCounts},
  {"l1t::CondHfRingEtSums", l1t::CondHfRingEtSums},
  {"l1t::CondBptx", l1t::CondBptx},
  {"l1t::CondExternal", l1t::CondExternal},
  {0, (l1t::GtConditionCategory) - 1}
};

}
// l1t::L1GtBoardType
l1t::L1GtBoardType l1t::l1GtBoardTypeStringToEnum(const std::string& label) {
    l1t::L1GtBoardType value = keyToValue(label.c_str(), l1GtBoardTypeStringToEnumMap);
    if (value == (l1t::L1GtBoardType) - 1) {
        edm::LogInfo("GtDefinitions") << "\n  '" << label
                << "' is not a recognized l1t::L1GtBoardType. \n  Return l1t::BoardNull.";
        value = l1t::BoardNull;
    }

    if (value == l1t::BoardNull) {
        edm::LogInfo("GtDefinitions")
                << "\n  l1t::BoardNull means no valid board type defined!";
    }

    return value;
}

std::string l1t::l1GtBoardTypeEnumToString(const l1t::L1GtBoardType& boardType) {
    char const *result= valueToKey(boardType, l1GtBoardTypeStringToEnumMap);
    if (boardType == l1t::BoardNull) {
        edm::LogInfo("GtDefinitions")
                << "\n  l1t::BoardNull means no valid board type defined!";
    }
    if (!result) {
      edm::LogInfo("GtDefinitions") << "\n  '" << boardType
                  << "' is not a recognized l1t::L1GtBoardType. "
                  << "\n  Return l1t::BoardNull, which means no valid board type defined!";
      return "l1t::BoardNull";
    }
    return result;
}


// l1t::L1GtPsbQuad

l1t::L1GtPsbQuad l1t::l1GtPsbQuadStringToEnum(const std::string& label) {
    l1t::L1GtPsbQuad value = keyToValue(label.c_str(), l1GtPsbQuadStringToEnumMap);
    // in case of unrecognized l1t::L1GtPsbQuad, return l1t::PsbQuadNull
    // to be dealt by the corresponding module
    if (value == -1) {
        edm::LogInfo("GtDefinitions") << "\n  '" << label
                << "' is not a recognized l1t::L1GtPsbQuad. \n  Return l1t::PsbQuadNull.";
        value = l1t::PsbQuadNull;
    }

    if (value == l1t::PsbQuadNull) {
        edm::LogInfo("GtDefinitions")
                << "\n  l1t::PsbQuadNull means no valid PSB quadruplet defined!";
    }

    return value;
}

std::string l1t::l1GtPsbQuadEnumToString(const l1t::L1GtPsbQuad& psbQuad) {
  char const*result = valueToKey(psbQuad, l1GtPsbQuadStringToEnumMap);
  if (psbQuad == l1t::PsbQuadNull)
    edm::LogInfo("GtDefinitions") << "\n  l1t::PsbQuadNull means no valid PSB quadruplet defined!";
  if (!result) {
    result = "l1t::PsbQuadNull";
    edm::LogInfo("GtDefinitions") << "\n  '" << psbQuad
                 << "' is not a recognized l1t::L1GtPsbQuad. "
                 << "\n  Return l1t::PsbQuadNull, which means no valid PSB quadruplet defined!";
  }

  return result;
}


l1t::GtConditionType l1t::l1GtConditionTypeStringToEnum(const std::string& label) {
    l1t::GtConditionType value = keyToValue(label.c_str(), l1GtConditionTypeStringToEnumMap);

    // in case of unrecognized l1t::GtConditionType, return l1t::TypeNull
    // to be dealt by the corresponding module
    if (value == (l1t::GtConditionType) -1) {
        edm::LogInfo("GtDefinitions")  << "\n  '" << label
                << "' is not a recognized l1t::GtConditionType. \n  Return l1t::TypeNull.";

        value = l1t::TypeNull;
    }

    if (value == l1t::TypeNull) {
        edm::LogInfo("GtDefinitions")
                << "\n  l1t::TypeNull means no valid condition type defined!";
    }

    return value;
}

std::string l1t::l1GtConditionTypeEnumToString(const l1t::GtConditionType& conditionType) {
  const char *result = valueToKey(conditionType, l1GtConditionTypeStringToEnumMap);
  if (conditionType == l1t::TypeNull)
    edm::LogInfo("GtDefinitions") 
      << "\n  Return l1t::TypeNull, which means no valid condition type defined!";
  if (!result) {
    result = "l1t::TypeNull";
    edm::LogInfo("GtDefinitions") << "\n  '" << conditionType
            << "' is not a recognized l1t::GtConditionType. "
            << "\n  Return l1t::TypeNull, which means no valid condition type defined!";
  }
  return result;
}

l1t::GtConditionCategory l1t::l1GtConditionCategoryStringToEnum(const std::string& label) {
  l1t::GtConditionCategory value = keyToValue(label.c_str(), l1GtConditionCategoryStringToEnumMap);
  // in case of unrecognized l1t::GtConditionCategory, return l1t::CondNull
  // to be dealt by the corresponding module
  if (value == (l1t::GtConditionCategory) -1) {
    edm::LogInfo("GtDefinitions") << "\n  '" << label
            << "' is not a recognized l1t::GtConditionCategory. \n  Return l1t::CondNull.";

    value = l1t::CondNull;
  }

  if (value == l1t::CondNull) {
      edm::LogInfo("GtDefinitions")
              << "\n  l1t::CondNull means no valid condition category defined!";
  }

  return value;
}

std::string l1t::l1GtConditionCategoryEnumToString(const l1t::GtConditionCategory& conditionCategory) {
  char const *result = valueToKey(conditionCategory, l1GtConditionCategoryStringToEnumMap);
  if (conditionCategory == l1t::CondNull)
    edm::LogInfo("GtDefinitions")
            << "\n  Return l1t::CondNull, which means no valid condition category defined!";
    
  if (!result) {
    result = "l1t::CondNull";
    edm::LogInfo("GtDefinitions") << "\n  '" << conditionCategory
            << "' is not a recognized l1t::GtConditionCategory. "
            << "\n  Return l1t::CondNull, which means no valid condition category defined!";
  }

  return result;
}
