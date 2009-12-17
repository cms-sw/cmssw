#ifndef CondFormats_L1TObjects_L1GtFwd_h
#define CondFormats_L1TObjects_L1GtFwd_h

/**
 *
 *
 * Description: enums for the L1 GT.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

/// board types in GT
enum L1GtBoardType { GTFE, FDL, PSB, GMT, TCS, TIM };

/// quadruples sent to GT via PSB
enum L1GtPsbQuad {Free,    TechTr,
                  IsoEGQ,  NoIsoEGQ,
                  CenJetQ, ForJetQ, TauJetQ,
                  ESumsQ,
                  JetCountsQ,
                  MQB1,    MQB2,     MQF3,       MQF4,
                  MQB5,    MQB6,     MQF7,       MQF8,
                  MQB9,    MQB10,    MQF11,      MQF12,
                  CastorQ,
                  HfQ,
                  BptxQ,
                  GtExternalQ};


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
enum L1GtConditionType { TypeNull,
                         Type1s, Type2s, Type2wsc, Type2cor, Type3s, Type4s,
                         TypeETM, TypeETT, TypeHTT, TypeHTM,
                         TypeJetCounts,
                         TypeCastor,
                         TypeHfBitCounts,
                         TypeHfRingEtSums,
                         TypeBptx,
                         TypeExternal};

/// condition categories
enum L1GtConditionCategory { CondNull,
                             CondMuon, CondCalo,
                             CondEnergySum, CondJetCounts,
                             CondCorrelation,
                             CondCastor,
                             CondHfBitCounts,
                             CondHfRingEtSums,
                             CondBptx,
                             CondExternal};

#endif /*CondFormats_L1TObjects_L1GtBoardMaps_h*/
