// Last commit: $Id: ConstantsForPresentation.h,v 1.4 2009/02/10 21:45:54 lowette Exp $

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
  
  static const char unknownPresentation_[]   = "UnknownPresentation";
  static const char undefinedPresentation_[] = "UndefinedPresentation";
  
  static const char histo1d_[]        = "Histo1D";
  static const char histo2dSum_[]     = "Histo2DSum";
  static const char histo2dScatter_[] = "Histo2DScatter";
  static const char profile1D_[]      = "Profile1D";

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


