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

/// GCT quadruples sent to GT
enum L1GtCaloQuad { NoIsoEGQ, IsoEGQ, CenJetQ, ForJetQ, TauJetQ, ESumsQ, JetCountsQ };

/// condition types
/// 1_s :   one particle
/// 2_s :   two particles, same type, no spatial correlations among them
/// 2_wsc : two particles, same type, with spatial correlations among them
/// 2_cor : two particles, different type, with spatial correlations among them
/// 3_s : three particles, same type
/// 4_s : four particles, same type
enum L1GtConditionType { Type1s, Type2s, Type2wsc, Type2cor, Type3s, Type4s };

/// condition categories
enum L1GtConditionCategory { MuonCond, CaloCond, EnergySumCond, JetCountsCond, CorrelationCond};

#endif /*CondFormats_L1TObjects_L1GtBoardMaps_h*/
