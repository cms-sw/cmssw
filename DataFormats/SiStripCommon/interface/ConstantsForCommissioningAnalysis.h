// Last commit: $Id: ConstantsForCommissioningAnalysis.h,v 1.10 2008/02/19 21:05:26 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H
#define DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H

#include <string>

/**
   @file ConstantsForCommissioningAnalysis.h
   @brief Various generic constants used by commissioning analysis
*/

namespace sistrip { 
  
  // ---------- General ----------

  static const std::string numberOfHistos_      = "UnexpectedNumberOfHistograms";
  static const std::string nullPtr_             = "NullPointerToHistogram";
  static const std::string numberOfBins_        = "UnexpectedNumberOfHistogramBins";
  static const std::string unexpectedBinNumber_ = "UnexpectedBinNumber";
  static const std::string noEntries_           = "NoEntriesInHistogramBin";
  static const std::string unexpectedTask_      = "UnexpectedTaskInHistoTitle";
  static const std::string unexpectedExtraInfo_ = "UnexpectedExtraInfoInHistoTitle";
  
  // ---------- Fast FED cabling ----------
  
  static const std::string fastCablingAnalysis_ = "FastCablingAnalysis";
  static const std::string invalidLightLevel_   = "InvalidLightLevel";
  static const std::string invalidTrimDacLevel_ = "InvalidTrimDacLevel";

  // ---------- FED cabling ----------
  
  static const std::string fedCablingAnalysis_ = "FedCablingAnalysis";
  static const std::string noCandidates_       = "NoChannelsAboveThreshold";
  
  // ---------- APV timing ----------
  
  static const std::string apvTimingAnalysis_   = "ApvTimingAnalysis";
  static const std::string smallDataRange_      = "SmallRangeInRawData";
  static const std::string smallTickMarkHeight_ = "SmallTickMarkHeight";
  static const std::string missingTickMark_     = "TickMarkNotFound";
  static const std::string tickMarkBelowThresh_ = "TickMarkHeightBelowThreshold";
  static const std::string noRisingEdges_       = "NoCandidateRisingEdges";
  static const std::string rejectedCandidate_   = "RejectedTickMarkCandidate";
  static const std::string incompletePlateau_   = "IncompletePlateau";
  static const std::string invalidRefTime_      = "InvalidRefTime";
  static const std::string invalidDelayTime_    = "InvalidDelayTime";

  // ---------- Opto scan ----------
  
  static const std::string optoScanAnalysis_      = "OptoScanAnalysis";
  static const std::string invalidZeroLightLevel_ = "InvalidZeroLightLevel";

  // ---------- VPSP scan ----------

  static const std::string noTopPlateau_    = "CannotFindTopPlateau";
  static const std::string noBottomPlateau_ = "CannotFindBottomPlateau";
  static const std::string noVpspSetting_   = "InvalidZeroLightLevel";
  static const std::string noBaselineLevel_ = "InvalidZeroLightLevel";

}

#endif // DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H


