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
 * \author: Vladimir Rekovic,   Brian Winer, OSU   Vasile Mihai Ghete - HEPHY Vienna
 *          Elisa Fontanesi - extended for three-body correlation conditions
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
  enum L1GtBoardType { MP7, BoardNull, L1GtBoardTypeInvalid = -1 };

  struct L1GtBoardTypeStringToEnum {
    const char* label;
    L1GtBoardType value;
  };

  L1GtBoardType l1GtBoardTypeStringToEnum(const std::string&);
  std::string l1GtBoardTypeEnumToString(const L1GtBoardType&);

  /// condition types
  /// TypeNull:  null type - for condition constructor only
  /// Type1s :   one particle
  /// Type2s :   two particles, same type, no spatial correlations among them
  /// Type2wsc : two particles, same type, with spatial correlations among them
  /// Type2cor : two particles, different type, with spatial correlations among them
  /// Type3s : three particles, same type
  /// Type4s : four particles, same type
  /// TypeETM, TypeETT, TypeHTT, TypeHTM  : ETM, ETT, HTT, HTM
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
    TypeETMHF,
    TypeTowerCount,
    TypeMinBiasHFP0,
    TypeMinBiasHFM0,
    TypeMinBiasHFP1,
    TypeMinBiasHFM1,
    TypeZDCP,
    TypeZDCM,
    TypeETTem,
    TypeExternal,
    Type2corWithOverlapRemoval,
    TypeCent0,
    TypeCent1,
    TypeCent2,
    TypeCent3,
    TypeCent4,
    TypeCent5,
    TypeCent6,
    TypeCent7,
    TypeAsymEt,
    TypeAsymHt,
    TypeAsymEtHF,
    TypeAsymHtHF,
    GtConditionTypeInvalid = -1
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
    CondZdcEnergySum,
    CondCorrelation,
    CondExternal,
    CondCorrelationWithOverlapRemoval,
    CondCorrelationThreeBody,
    CondMuonShower,
    GtConditionCategoryInvalid = -1
  };

  struct GtConditionCategoryStringToEnum {
    const char* label;
    GtConditionCategory value;
  };

  GtConditionCategory l1GtConditionCategoryStringToEnum(const std::string&);
  std::string l1GtConditionCategoryEnumToString(const GtConditionCategory&);

}  // namespace l1t
#endif
