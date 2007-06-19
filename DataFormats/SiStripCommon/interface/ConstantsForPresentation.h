// Last commit: $Id: ConstantsForPresentation.h,v 1.2 2007/03/21 08:22:59 bainbrid Exp $

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
  
  static const std::string histo1d_        = "Histo1D";
  static const std::string histo2dSum_     = "Histo2DSum";
  static const std::string histo2dScatter_ = "Histo2DScatter";
  static const std::string profile1D_      = "Profile1D";

  // ---------- Enumerated type ---------- 

  enum Presentation { UNKNOWN_PRESENTATION   = sistrip::unknown_, 
		      UNDEFINED_PRESENTATION = sistrip::invalid_, 
		      HISTO_1D         = 1,
		      HISTO_2D_SUM     = 2,
		      HISTO_2D_SCATTER = 3,
		      PROFILE_1D       = 4
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForPresentation_H


