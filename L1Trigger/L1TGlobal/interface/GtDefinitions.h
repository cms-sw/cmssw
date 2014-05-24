#ifndef L1Trigger_L1TGlobal_GtDefinitions_h
#define L1Trigger_L1TGlobal_GtDefinitions_h

/**
 *
 *
 * Description: enums for the L1 GT.
 *
 * Implementation:
 *    Defines various enums for CondFormats L1 GT. For each enum, define the
 *    lightweight "maps" for enum string label and enum value
 *
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

// user include files

namespace l1t {
/// board types in GT
enum L1GtBoardType {
    GTFE,
    FDL,
    PSB,
    GMT,
    TCS,
    TIM,
    BoardNull
};

struct L1GtBoardTypeStringToEnum {
    const char* label;
    L1GtBoardType value;
};

L1GtBoardType l1GtBoardTypeStringToEnum(const std::string&);
std::string l1GtBoardTypeEnumToString(const L1GtBoardType&);

/// quadruples sent to GT via PSB
enum L1GtPsbQuad {
    Free,
    TechTr,
    IsoEGQ,
    NoIsoEGQ,
    CenJetQ,
    ForJetQ,
    TauJetQ,
    ESumsQ,
    JetCountsQ,
    MQB1,
    MQB2,
    MQF3,
    MQF4,
    MQB5,
    MQB6,
    MQF7,
    MQF8,
    MQB9,
    MQB10,
    MQF11,
    MQF12,
    CastorQ,
    HfQ,
    BptxQ,
    GtExternalQ,
    PsbQuadNull
};

struct L1GtPsbQuadStringToEnum {
    const char* label;
    L1GtPsbQuad value;
};

L1GtPsbQuad l1GtPsbQuadStringToEnum(const std::string&);
std::string l1GtPsbQuadEnumToString(const L1GtPsbQuad&);

/// condition types
/// TypeNull:  null type - for condition constructor only
/// Type1s :   one particle
/// Type2s :   two particles, same type, no spatial correlations among them
/// Type2wsc : two particles, same type, with spatial correlations among them
/// Type2cor : two particles, different type, with spatial correlations among them
/// Type3s : three particles, same type
/// Type4s : four particles, same type
/// TypeETM, TypeETT, TypeHTT, TypeHTM  : ETM, ETT, HTT, HTM
/// TypeJetCounts : JetCounts
/// TypeCastor : CASTOR condition (logical result only; definition in CASTOR)
/// TypeHfBitCounts :  HfBitCounts
/// TypeHfRingEtSums : HfRingEtSums
/// TypeBptx: BPTX (logical result only; definition in BPTX system)
/// TypeExternal: external conditions (logical result only; definition in L1 GT external systems)
enum GtConditionType {
    TypeNull,
    Type1s,
    Type2s,
    Type2wsc,
    Type2cor,
    Type3s,
    Type4s,
    TypeETM,
    TypeETT,
    TypeHTT,
    TypeHTM,
    TypeJetCounts,
    TypeCastor,
    TypeHfBitCounts,
    TypeHfRingEtSums,
    TypeBptx,
    TypeExternal
};

struct GtConditionTypeStringToEnum {
    const char* label;
    GtConditionType value;
};

GtConditionType l1GtConditionTypeStringToEnum(const std::string&);
std::string l1GtConditionTypeEnumToString(const GtConditionType&);

/// condition categories
enum GtConditionCategory {
    CondNull,
    CondMuon,
    CondCalo,
    CondEnergySum,
    CondJetCounts,
    CondCorrelation,
    CondCastor,
    CondHfBitCounts,
    CondHfRingEtSums,
    CondBptx,
    CondExternal
};

struct GtConditionCategoryStringToEnum {
    const char* label;
    GtConditionCategory value;
};

GtConditionCategory l1GtConditionCategoryStringToEnum(const std::string&);
std::string l1GtConditionCategoryEnumToString(const GtConditionCategory&);

}
#endif
