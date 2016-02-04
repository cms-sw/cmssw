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

// forward declarations

// L1GtBoardType
L1GtBoardType l1GtBoardTypeStringToEnum(const std::string& label) {

    static L1GtBoardTypeStringToEnum l1GtBoardTypeStringToEnumMap[] = {
            {"GTFE", GTFE},
            {"FDL", FDL},
            {"PSB", PSB},
            {"GMT", GMT},
            {"TCS", TCS},
            {"TIM", TIM},
            {"BoardNull", BoardNull},
            {0, (L1GtBoardType) - 1}
    };

    L1GtBoardType value = (L1GtBoardType) - 1;

    bool found = false;
    for (int i = 0; l1GtBoardTypeStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(), l1GtBoardTypeStringToEnumMap[i].label)) {
            found = true;
            value = l1GtBoardTypeStringToEnumMap[i].value;
        }

    // in case of unrecognized L1GtBoardType, return BoardNull
    // to be dealt by the corresponding module
    if (!found) {
        edm::LogInfo("L1GtDefinitions") << "\n  '" << label
                << "' is not a recognized L1GtBoardType. \n  Return BoardNull.";

        value = BoardNull;
    }

    if (value == BoardNull) {
        edm::LogInfo("L1GtDefinitions")
                << "\n  BoardNull means no valid board type defined!";
    }

    return value;
}

std::string l1GtBoardTypeEnumToString(const L1GtBoardType& boardType) {

    std::string boardTypeString;

    switch (boardType) {

        case GTFE: {
            boardTypeString = "GTFE";
        }
            break;
        case FDL: {
            boardTypeString = "FDL";
        }
            break;
        case PSB: {
            boardTypeString = "PSB";
        }
            break;
        case GMT: {
            boardTypeString = "GMT";
        }
            break;
        case TCS: {
            boardTypeString = "TCS";
        }
            break;
        case TIM: {
            boardTypeString = "TIM";
        }
            break;
        case BoardNull: {
            boardTypeString = "BoardNull";
            edm::LogInfo("L1GtDefinitions")
                    << "\n  BoardNull means no valid board type defined!";
        }
            break;
        default: {
            boardTypeString = "BoardNull";
            edm::LogInfo("L1GtDefinitions") << "\n  '" << boardType
                    << "' is not a recognized L1GtBoardType. "
                    << "\n  Return BoardNull, which means no valid board type defined!";

        }
            break;
    }

    return boardTypeString;

}


// L1GtPsbQuad

L1GtPsbQuad l1GtPsbQuadStringToEnum(const std::string& label) {

    static L1GtPsbQuadStringToEnum l1GtPsbQuadStringToEnumMap[] = {
            {"Free", Free},
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
            {0, (L1GtPsbQuad) - 1}
    };

    L1GtPsbQuad value = (L1GtPsbQuad) - 1;

    bool found = false;
    for (int i = 0; l1GtPsbQuadStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(), l1GtPsbQuadStringToEnumMap[i].label)) {
            found = true;
            value = l1GtPsbQuadStringToEnumMap[i].value;
        }

    // in case of unrecognized L1GtPsbQuad, return PsbQuadNull
    // to be dealt by the corresponding module
    if (!found) {
        edm::LogInfo("L1GtDefinitions") << "\n  '" << label
                << "' is not a recognized L1GtPsbQuad. \n  Return PsbQuadNull.";

        value = PsbQuadNull;
    }

    if (value == PsbQuadNull) {
        edm::LogInfo("L1GtDefinitions")
                << "\n  PsbQuadNull means no valid PSB quadruplet defined!";
    }

    return value;
}

std::string l1GtPsbQuadEnumToString(const L1GtPsbQuad& psbQuad) {

    std::string psbQuadString;

    switch (psbQuad) {

        case Free: {
            psbQuadString = "Free";
        }
            break;
        case TechTr: {
            psbQuadString = "TechTr";
        }
            break;
        case IsoEGQ: {
            psbQuadString = "IsoEGQ";
        }
            break;
        case NoIsoEGQ: {
            psbQuadString = "NoIsoEGQ";
        }
            break;
        case CenJetQ: {
            psbQuadString = "CenJetQ";
        }
            break;
        case ForJetQ: {
            psbQuadString = "ForJetQ";
        }
            break;
        case TauJetQ: {
            psbQuadString = "TauJetQ";
        }
            break;
        case ESumsQ: {
            psbQuadString = "ESumsQ";
        }
            break;
        case JetCountsQ: {
            psbQuadString = "JetCountsQ";
        }
            break;
        case MQB1: {
            psbQuadString = "MQB1";
        }
            break;
        case MQB2: {
            psbQuadString = "MQB2";
        }
            break;
        case MQF3: {
            psbQuadString = "MQF3";
        }
            break;
        case MQF4: {
            psbQuadString = "MQF4";
        }
            break;
        case MQB5: {
            psbQuadString = "MQB5";
        }
            break;
        case MQB6: {
            psbQuadString = "MQB6";
        }
            break;
        case MQF7: {
            psbQuadString = "MQF7";
        }
            break;
        case MQF8: {
            psbQuadString = "MQF9";
        }
            break;
        case MQB9: {
            psbQuadString = "MQB9";
        }
            break;
        case MQB10: {
            psbQuadString = "MQB10";
        }
            break;
        case MQF11: {
            psbQuadString = "MQF11";
        }
            break;
        case MQF12: {
            psbQuadString = "MQF12";
        }
            break;
        case CastorQ: {
            psbQuadString = "CastorQ";
        }
            break;
        case HfQ: {
            psbQuadString = "HfQ";
        }
            break;
        case BptxQ: {
            psbQuadString = "BptxQ";
        }
            break;
        case GtExternalQ: {
            psbQuadString = "GtExternalQ";
        }
            break;
        case PsbQuadNull: {
            psbQuadString = "PsbQuadNull";
            edm::LogInfo("L1GtDefinitions")
                    << "\n  PsbQuadNull means no valid PSB quadruplet defined!";
        }
            break;
        default: {
            psbQuadString = "PsbQuadNull";
            edm::LogInfo("L1GtDefinitions") << "\n  '" << psbQuad
                    << "' is not a recognized L1GtPsbQuad. "
                    << "\n  Return PsbQuadNull, which means no valid PSB quadruplet defined!";

        }
            break;
    }

    return psbQuadString;

}

// L1GtConditionType

L1GtConditionType l1GtConditionTypeStringToEnum(const std::string& label) {

    static L1GtConditionTypeStringToEnum l1GtConditionTypeStringToEnumMap[] = {
            {"TypeNull", TypeNull},
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
            {0, (L1GtConditionType) - 1}
    };

    L1GtConditionType value = (L1GtConditionType) - 1;

    bool found = false;
    for (int i = 0; l1GtConditionTypeStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(), l1GtConditionTypeStringToEnumMap[i].label)) {
            found = true;
            value = l1GtConditionTypeStringToEnumMap[i].value;
        }

    // in case of unrecognized L1GtConditionType, return TypeNull
    // to be dealt by the corresponding module
    if (!found) {
        edm::LogInfo("L1GtDefinitions")  << "\n  '" << label
                << "' is not a recognized L1GtConditionType. \n  Return TypeNull.";

        value = TypeNull;
    }

    if (value == TypeNull) {
        edm::LogInfo("L1GtDefinitions")
                << "\n  TypeNull means no valid condition type defined!";
    }

    return value;
}

std::string l1GtConditionTypeEnumToString(
        const L1GtConditionType& conditionType) {

    std::string conditionTypeString;

    switch (conditionType) {

        case TypeNull: {
            conditionTypeString = "TypeNull";
            edm::LogInfo("L1GtDefinitions")
                    << "\n  TypeNull means no valid condition type defined!";
        }

            break;
        case Type1s: {
            conditionTypeString = "Type1s";
        }

            break;
        case Type2s: {
            conditionTypeString = "Type2s";
        }

            break;
        case Type2wsc: {
            conditionTypeString = "Type2wsc";
        }

            break;
        case Type2cor: {
            conditionTypeString = "Type2cor";
        }

            break;
        case Type3s: {
            conditionTypeString = "Type3s";
        }

            break;
        case Type4s: {
            conditionTypeString = "Type4s";
        }

            break;
        case TypeETM: {
            conditionTypeString = "TypeETM";
        }

            break;
        case TypeETT: {
            conditionTypeString = "TypeETT";
        }

            break;
        case TypeHTT: {
            conditionTypeString = "TypeHTT";
        }

            break;
        case TypeHTM: {
            conditionTypeString = "TypeHTM";
        }

            break;
        case TypeJetCounts: {
            conditionTypeString = "TypeJetCounts";
        }

            break;
        case TypeCastor: {
            conditionTypeString = "TypeCastor";
        }

            break;
        case TypeHfBitCounts: {
            conditionTypeString = "TypeHfBitCounts";
        }

            break;
        case TypeHfRingEtSums: {
            conditionTypeString = "TypeHfRingEtSums";
        }

            break;
        case TypeBptx: {
            conditionTypeString = "TypeBptx";
        }

            break;
        case TypeExternal: {
            conditionTypeString = "TypeExternal";
        }

            break;
        default: {
            conditionTypeString = "TypeNull";
            edm::LogInfo("L1GtDefinitions") << "\n  '" << conditionType
                    << "' is not a recognized L1GtConditionType. "
                    << "\n  Return TypeNull, which means no valid condition type defined!";
        }
            break;
    }

    return conditionTypeString;
}

// L1GtConditionCategory

L1GtConditionCategory l1GtConditionCategoryStringToEnum(const std::string& label) {

    static L1GtConditionCategoryStringToEnum l1GtConditionCategoryStringToEnumMap[] = {
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
            {0, (L1GtConditionCategory) - 1}
    };

    L1GtConditionCategory value = (L1GtConditionCategory) - 1;

    bool found = false;
    for (int i = 0; l1GtConditionCategoryStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(),
                l1GtConditionCategoryStringToEnumMap[i].label)) {
            found = true;
            value = l1GtConditionCategoryStringToEnumMap[i].value;
        }

    // in case of unrecognized L1GtConditionCategory, return CondNull
    // to be dealt by the corresponding module
    if (!found) {
        edm::LogInfo("L1GtDefinitions") << "\n  '" << label
                << "' is not a recognized L1GtConditionCategory. \n  Return CondNull.";

        value = CondNull;
    }

    if (value == CondNull) {
        edm::LogInfo("L1GtDefinitions")
                << "\n  CondNull means no valid condition category defined!";
    }

    return value;
}

std::string l1GtConditionCategoryEnumToString(
        const L1GtConditionCategory& conditionCategory) {

    std::string conditionCategoryString;

    switch (conditionCategory) {

        case CondNull: {
            conditionCategoryString = "CondNull";
            edm::LogInfo("L1GtDefinitions")
                    << "\n  CondNull means no valid condition category defined!";
        }

            break;
        case CondMuon: {
            conditionCategoryString = "CondMuon";
        }

            break;
        case CondCalo: {
            conditionCategoryString = "CondCalo";
        }

            break;
        case CondEnergySum: {
            conditionCategoryString = "CondEnergySum";
        }

            break;
        case CondJetCounts: {
            conditionCategoryString = "CondJetCounts";
        }

            break;
        case CondCorrelation: {
            conditionCategoryString = "CondCorrelation";
        }

            break;
        case CondCastor: {
            conditionCategoryString = "CondCastor";
        }

            break;
        case CondHfBitCounts: {
            conditionCategoryString = "CondHfBitCounts";
        }

            break;
        case CondHfRingEtSums: {
            conditionCategoryString = "CondHfRingEtSums";
        }

            break;
        case CondBptx: {
            conditionCategoryString = "CondBptx";
        }

            break;
        case CondExternal: {
            conditionCategoryString = "CondExternal";
        }

            break;
        default: {
            conditionCategoryString = "CondNull";
            edm::LogInfo("L1GtDefinitions") << "\n  '" << conditionCategory
                    << "' is not a recognized L1GtConditionCategory. "
                    << "\n  Return CondNull, which means no valid condition category defined!";

        }
            break;
    }

    return conditionCategoryString;
}

