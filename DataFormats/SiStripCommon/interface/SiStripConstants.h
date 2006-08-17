#ifndef DataFormats_SiStripCommon_SiStripConstants_H
#define DataFormats_SiStripCommon_SiStripConstants_H

#include "boost/cstdint.hpp"
#include <string>

// -----------------------------------------------------------------------------
// Some Generic constants
namespace sistrip { 
  
  // Invalid value
  static const uint16_t invalid_ = 0xFFFF;

}

// -----------------------------------------------------------------------------
// Useful constants associated with the FED
namespace sistrip { 

  // FED = Front-End Driver, FEUNIT = Front-End Unit, FEDCH = Fed Channel
  static const uint16_t FEDCH_PER_FEUNIT = 12;
  static const uint16_t FEUNITS_PER_FED  = 8;
  static const uint16_t FEDCH_PER_FED    = FEDCH_PER_FEUNIT * FEUNITS_PER_FED; // 96
  
  // APV = APV25 front-end readout chip
  static const uint16_t APVS_PER_FEDCH   = 2;
  static const uint16_t APVS_PER_FEUNIT  = APVS_PER_FEDCH * FEDCH_PER_FEUNIT; // 24
  static const uint16_t APVS_PER_FED     = APVS_PER_FEUNIT * FEUNITS_PER_FED; // 194
  
  // STRIP = Detector strips 
  static const uint16_t STRIPS_PER_APV    = 128;
  static const uint16_t STRIPS_PER_FEDCH  = STRIPS_PER_APV * APVS_PER_FEDCH;
  static const uint16_t STRIPS_PER_FEUNIT = STRIPS_PER_FEDCH * FEDCH_PER_FEUNIT; // 3072
  static const uint16_t STRIPS_PER_FED    = STRIPS_PER_FEUNIT * FEUNITS_PER_FED; // 24576

  // Constants related to FED buffer formatting
  static const uint16_t DAQ_HDR_SIZE        = 8;
  static const uint16_t TRK_HDR_SIZE        = 8;
  static const uint16_t FE_HDR_SIZE         = 16;
  static const uint16_t APV_ERROR_HDR_SIZE  = 24;
  static const uint16_t FULL_DEBUG_HDR_SIZE = 8 * FE_HDR_SIZE;
  
}

// -----------------------------------------------------------------------------
// Constants related to error conditions
namespace sistrip { 

  static const std::string sistrip_ = "SiStrip|";
  
  // Error categories
  static const std::string DQM_FWK    = sistrip_ + "DQM_FWK";
  static const std::string DQM_HISTOS = sistrip_ + "DQM_HISTOS";
  
}
  
#endif // DataFormats_SiStripCommon_SiStripConstants_H


