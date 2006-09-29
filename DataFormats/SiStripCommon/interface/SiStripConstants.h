#ifndef DataFormats_SiStripCommon_SiStripConstants_H
#define DataFormats_SiStripCommon_SiStripConstants_H

#include "boost/cstdint.hpp"
#include <string>

// -----------------------------------------------------------------------------
// Constants related to error conditions
namespace sistrip { 

  static const std::string sistrip_ = "SiStrip|";
  
  // Error categories
  static const std::string DQM_FWK    = sistrip_ + "DQM_FWK";
  static const std::string DQM_HISTOS = sistrip_ + "DQM_HISTOS";
  
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

  // 
  static const uint16_t DAQ_HDR_SIZE        = 8;
  static const uint16_t TRK_HDR_SIZE        = 8;
  static const uint16_t FE_HDR_SIZE         = 16;
  static const uint16_t APV_ERROR_HDR_SIZE  = 24;
  static const uint16_t FULL_DEBUG_HDR_SIZE = 8 * FE_HDR_SIZE;
  
}

// -----------------------------------------------------------------------------
// Constants associated with the naming of DQM histograms
namespace sistrip { 
  
  // generic constants
  static const uint16_t all_     = 0xFFFF;
  static const std::string root_ = "SiStrip";
  static const std::string dir_  = "/";
  static const std::string sep_  = "_";
  static const std::string commissioningTask_ = "SiStripCommissioningTask";
  
  // views
  static const std::string controlView_  = "ControlView";
  static const std::string readoutView_  = "ReadoutView";
  static const std::string detectorView_ = "DetectorView";
  static const std::string unknownView_  = "UnknownView";

  // control and readout parameters
  static const std::string fecCrate_    = "FecCrate";
  static const std::string fecSlot_     = "FecSlot";
  static const std::string fecRing_     = "FecRing";
  static const std::string ccuAddr_     = "CcuAddr";
  static const std::string ccuChan_     = "CcuChan";
  static const std::string fedId_       = "FedId";
  static const std::string fedChannel_  = "FedChannel";

  // commissioning task
  static const std::string fedCabling_  = "FedCabling";
  static const std::string apvTiming_   = "ApvTiming";
  static const std::string fedTiming_   = "FedTiming";
  static const std::string optoScan_    = "OptoScan";
  static const std::string vpspScan_    = "VpspScan";
  static const std::string pedestals_   = "Pedestals";
  static const std::string apvLatency_  = "ApvLatency";
  static const std::string unknownTask_ = "UnknownTask";

  // histo contents
  static const std::string sum2_            = "SumOfSquares";
  static const std::string sum_             = "SumOfContents";
  static const std::string num_             = "NumOfEntries";
  static const std::string unknownContents_ = "UnknownContents";

  // key 
  static const std::string fedKey_     = "FedKey";
  static const std::string fecKey_     = "FecKey";
  static const std::string detKey_     = "DetId"; //@@ necessary?
  static const std::string unknownKey_ = "UnknownKey";

  // granularity
  static const std::string lldChan_            = "LldChan";
  static const std::string apvPair_            = "ApvPair";
  static const std::string apv_                = "Apv";
  static const std::string unknownGranularity_ = "UnknownGranularity";

  // extra histogram information 
  static const std::string gain_              = "Gain";
  static const std::string digital_           = "Digital";
  static const std::string pedsAndRawNoise_   = "PedsAndRawNoise";
  static const std::string residualsAndNoise_ = "ResidualsAndNoise";
  static const std::string commonMode_        = "CommonMode";
  
}

#endif // DataFormats_SiStripCommon_SiStripConstants_H


