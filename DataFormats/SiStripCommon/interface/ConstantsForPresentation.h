// Last commit: $Id: $

#ifndef DataFormats_SiStripCommon_ConstantsForPresentation_H
#define DataFormats_SiStripCommon_ConstantsForPresentation_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForPresentation.h

    @brief Constants and enumerated type for sistrip::Presentation
*/

namespace sistrip { 

  // ---------- Constants ---------- 
  
  static const std::string unknownPresentation_   = "UnknownPresentation";
  static const std::string undefinedPresentation_ = "UndefinedPresentation";
  
  static const std::string summaryHisto_ = "SummaryHisto";
  static const std::string summary1D_    = "Summary1D";
  static const std::string summary2D_    = "Summary2D";
  static const std::string summaryProf_  = "SummaryProfile";

  // ---------- Enumerated type ---------- 

  enum Presentation { UNKNOWN_PRESENTATION   = sistrip::unknown_, 
		      UNDEFINED_PRESENTATION = sistrip::invalid_, 
		      SUMMARY_HISTO = 1,
		      SUMMARY_1D    = 2,
		      SUMMARY_2D    = 3,
		      SUMMARY_PROF  = 4
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForPresentation_H


