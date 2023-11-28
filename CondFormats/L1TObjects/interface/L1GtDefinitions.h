#ifndef CondFormats_L1TObjects_L1GtDefinitions_h
#define CondFormats_L1TObjects_L1GtDefinitions_h

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
#include "DataFormats/L1GlobalTrigger/interface/L1GtDefinitions.h"

/// board types in GT
enum L1GtBoardType { GTFE, FDL, PSB, GMT, TCS, TIM, BoardNull, L1GtBoardTypeInvalid = -1 };

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
  PsbQuadNull,
  L1GtPsbQuadInvalid = -1
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
/// Type2CorrWithOverlapRemoval: three particles, first two with spatial correlations among them, third used for removal if overlap

struct L1GtConditionTypeStringToEnum {
  const char* label;
  L1GtConditionType value;
};

L1GtConditionType l1GtConditionTypeStringToEnum(const std::string&);
std::string l1GtConditionTypeEnumToString(const L1GtConditionType&);

struct L1GtConditionCategoryStringToEnum {
  const char* label;
  L1GtConditionCategory value;
};

L1GtConditionCategory l1GtConditionCategoryStringToEnum(const std::string&);
std::string l1GtConditionCategoryEnumToString(const L1GtConditionCategory&);

#endif
