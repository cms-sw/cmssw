// Last commit: $Id: ConstantsForHistoType.h,v 1.1 2007/06/19 12:16:52 bainbrid Exp $

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
  
  static const char unknownHistoType_[]   = "UnknownHistoType";
  static const char undefinedHistoType_[] = "UndefinedHistoType";
  
  static const char expertHisto_[]  = "ExpertHisto";
  static const char summaryHisto_[] = "SummaryHisto";

  // ---------- Enumerated type ---------- 

  enum HistoType { UNKNOWN_HISTO_TYPE   = sistrip::unknown_, 
		   UNDEFINED_HISTO_TYPE = sistrip::invalid_, 
		   EXPERT_HISTO  = 1,
		   SUMMARY_HISTO = 2
  };
  
}
  
#endif // DataFormats_SiStripCommon_ConstantsForHistoType_H


