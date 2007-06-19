// Last commit: $Id: ConstantsForHistoType.h,v 1.2 2007/03/21 08:22:59 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForHistoType_H
#define DataFormats_SiStripCommon_ConstantsForHistoType_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForHistoType.h

    @brief Constants and enumerated type for sistrip::Presentation
*/

namespace sistrip { 

  // ---------- Constants ---------- 
  
  static const std::string unknownHistoType_   = "UnknownHistoType";
  static const std::string undefinedHistoType_ = "UndefinedHistoType";
  
  static const std::string expertHisto_  = "ExpertHisto";
  static const std::string summaryHisto_ = "SummaryHisto";

  // ---------- Enumerated type ---------- 

  enum HistoType { UNKNOWN_HISTO_TYPE   = sistrip::unknown_, 
		   UNDEFINED_HISTO_TYPE = sistrip::invalid_, 
		   EXPERT_HISTO  = 1,
		   SUMMARY_HISTO = 2
  };
  
}
  
#endif // DataFormats_SiStripCommon_ConstantsForHistoType_H


