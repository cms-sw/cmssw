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
 * $Date:$
 * $Revision:$
 *
 */

/// board types in GT
enum L1GtBoardType {GTFE, FDL, PSB, GMT, TCS, TIM};

/// GCT quadruples sent to GT
enum L1GtCaloQuad { NoIsoEGQ, IsoEGQ, CenJetQ, ForJetQ, TauJetQ, ESumsQ, JetCountsQ };


#endif /*CondFormats_L1TObjects_L1GtBoardMaps_h*/
