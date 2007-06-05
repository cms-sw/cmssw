// Last commit: $Id: $

#ifndef DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H
#define DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H

#include <string>

/**
   @file ConstantsForCommissioningAnalysis.h
   @brief Various generic constants used by commissioning analysis
*/

namespace sistrip { 
  
  // ---------- General ----------

  static const std::string nullPtr_   = "NullPointerToHistogram";
  static const std::string histoBins_ = "UnexpectedNumberOfHistogramBins";
  
  // ---------- APV timing ----------
  
  static const std::string apvTimingAnalysis_   = "ApvTimingAnalysis";
  static const std::string smallDataRange_      = "SmallRangeInRawData";
  static const std::string smallTickMarkHeight_ = "SmallTickMarkHeight";
  static const std::string missingTickMark_     = "TickMarkNotFound";
  static const std::string tickMarkBelowThresh_ = "TickMarkHeightBelowThreshold";

}

#endif // DataFormats_SiStripCommon_ConstantsForCommissioningAnalysis_H


