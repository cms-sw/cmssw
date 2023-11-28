#ifndef DataFormats_L1GlobalTrigger_L1GtDefinitions_h
#define DataFormats_L1GlobalTrigger_L1GtDefinitions_h

enum L1GtConditionType {
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
  TypeExternal,
  Type2corWithOverlapRemoval,
  L1GtConditionTypeInvalid = -1
};

/// condition categories
enum L1GtConditionCategory {
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
  CondExternal,
  CondCorrelationWithOverlapRemoval,
  L1GtConditionCategoryInvalid = -1
};

#endif
