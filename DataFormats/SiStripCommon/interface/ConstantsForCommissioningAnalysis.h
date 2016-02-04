// Last commit: $Id: ConstantsForCommissioningAnalysis.h,v 1.15 2010/06/02 09:40:31 wto Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H
#define DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H

#include <string>

/**
   @file ConstantsForCommissioningAnalysis.h
   @brief Various generic constants used by commissioning analysis
*/

namespace sistrip { 
  
  // ---------- General ----------

  static const char numberOfHistos_[]      = "UnexpectedNumberOfHistograms";
  static const char nullPtr_[]             = "NullPointerToHistogram";
  static const char numberOfBins_[]        = "UnexpectedNumberOfHistogramBins";
  static const char unexpectedBinNumber_[] = "UnexpectedBinNumber";
  static const char noEntries_[]           = "NoEntriesInHistogramBin";
  static const char unexpectedTask_[]      = "UnexpectedTaskInHistoTitle";
  static const char unexpectedExtraInfo_[] = "UnexpectedExtraInfoInHistoTitle";
  
  // ---------- Fast FED cabling ----------
  
  static const char fastCablingAnalysis_[] = "FastCablingAnalysis";
  static const char invalidLightLevel_[]   = "InvalidLightLevel";
  static const char invalidTrimDacLevel_[] = "InvalidTrimDacLevel";

  // ---------- FED cabling ----------
  
  static const char fedCablingAnalysis_[] = "FedCablingAnalysis";
  static const char noCandidates_[]       = "NoChannelsAboveThreshold";
  
  // ---------- APV timing ----------
  
  static const char apvTimingAnalysis_[]   = "ApvTimingAnalysis";
  static const char smallDataRange_[]      = "SmallRangeInRawData";
  static const char smallTickMarkHeight_[] = "SmallTickMarkHeight";
  static const char missingTickMark_[]     = "TickMarkNotFound";
  static const char tickMarkBelowThresh_[] = "TickMarkHeightBelowThreshold";
  static const char noRisingEdges_[]       = "NoCandidateRisingEdges";
  static const char rejectedCandidate_[]   = "RejectedTickMarkCandidate";
  static const char incompletePlateau_[]   = "IncompletePlateau";
  static const char invalidRefTime_[]      = "InvalidRefTime";
  static const char invalidDelayTime_[]    = "InvalidDelayTime";
  static const char tickMarkRecovered_[]   = "TickMarkRecovered";


  // ---------- Opto scan ----------
  
  static const char optoScanAnalysis_[]      = "OptoScanAnalysis";
  static const char invalidZeroLightLevel_[] = "InvalidZeroLightLevel";

  // ---------- VPSP scan ----------

  static const char noTopPlateau_[]    = "CannotFindTopPlateau";
  static const char noBottomPlateau_[] = "CannotFindBottomPlateau";
  static const char noVpspSetting_[]   = "InvalidZeroLightLevel";
  static const char noBaselineLevel_[] = "InvalidZeroLightLevel";

}

#endif // DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H


