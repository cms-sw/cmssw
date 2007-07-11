// Last commit: $Id: ConstantsForCommissioningAnalysis.h,v 1.4 2007/07/04 08:27:55 bainbrid Exp $

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
  static const std::string noEntries_           = "NoEntriesInHistogramBin";
  static const std::string unexpectedTask_      = "UnexpectedTaskInHistoTitle";
  static const std::string unexpectedExtraInfo_ = "UnexpectedExtraInfoInHistoTitle";
  
  // ---------- Fast FED cabling ----------
  
  static const std::string fastCablingAnalysis_ = "FastFedCablingAnalysis";

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

  // ---------- Opto scan ----------
  
  static const std::string optoScanAnalysis_ = "OptoScanAnalysis";

}

#endif // DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H


