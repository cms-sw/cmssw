#ifndef DataFormats_SiStripCommon_ConstantsForView_H
#define DataFormats_SiStripCommon_ConstantsForView_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForView.h

    @brief Constants and enumerated types for sistrip::View
*/

namespace sistrip { 

  // ---------- Constants ---------- 

  static const std::string unknownView_   = "UnknownView";
  static const std::string undefinedView_ = "UndefinedView";
  
  static const std::string readoutView_   = "ReadoutView";
  static const std::string controlView_   = "ControlView";
  static const std::string detectorView_  = "DetectorView";
  
  // ---------- Enumerated type ---------- 

  enum View { UNKNOWN_VIEW   = sistrip::unknown_, 
	      UNDEFINED_VIEW = sistrip::invalid_, 
	      READOUT_VIEW   = 1, 
	      CONTROL_VIEW   = 2, 
	      DETECTOR_VIEW  = 3 
  };
  
}
  
#endif // DataFormats_SiStripCommon_ConstantsForView_H


