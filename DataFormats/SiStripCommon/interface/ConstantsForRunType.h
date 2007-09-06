// Last commit: $Id: ConstantsForRunType.h,v 1.5 2007/07/11 11:13:59 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForRunType_H
#define DataFormats_SiStripCommon_ConstantsForRunType_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForRunType.h

    @brief Constants and enumerated type for sistrip::RunType
*/

namespace sistrip { 
  
  // ---------- Constants ---------- 
  
  static const std::string unknownRunType_   = "UnknownRunType";
  static const std::string undefinedRunType_ = "UndefinedRunType";
  
  static const std::string fastCabling_      = "FastCabling";
  static const std::string fedCabling_       = "FedCabling";
  static const std::string apvTiming_        = "ApvTiming";
  static const std::string fedTiming_        = "FedTiming";
  static const std::string optoScan_         = "OptoScan";
  static const std::string vpspScan_         = "VpspScan";
  static const std::string pedestals_        = "Pedestals";
  static const std::string apvLatency_       = "ApvLatency";
  static const std::string fineDelay_        = "FineDelay";
  static const std::string calibrationP_     = "CalibrationPeak";
  static const std::string calibrationD_     = "CalibrationDeco";
  static const std::string calibrationScanP_ = "CalibrationScanPeak";
  static const std::string calibrationScanD_ = "CalibrationScanDeco";
  static const std::string daqScopeMode_     = "DaqScopeMode";
  static const std::string physics_          = "Physics";
  
  // ---------- Enumerated type ---------- 
  
  /** 
   * Run types: (equivalent "TrackerSupervisor" enums in brackets): 
   * unknown run type,
   * undefined run type,
   * "fast" connection of FED channels to APV pairs (XTOFS_CONNECTION = 21), 
   * connection of FED channels to APV pairs (BARE_CONNECTION = 13), 
   * connection of FED channels to APV pairs (CONNECTION = 11),
   * connection of FED channels to APV pairs (FAST_CONNECTION = 16),
   * relative APV synchronisation (TIMING = 5), 
   * relative APV synchronisation using FED delays (TIMING_FED = 12), 
   * bias and gain scan for LLD device (GAINSCAN = 4), 
   * APV baseline scan (VPSPSCAN = 14), 
   * FED calibration run for pedestals and noise (PEDESTAL = 2), 
   * coarse (25ns) APV latency scan for beam (LATENCY = 6),
   * fine (1ns) PLL delay scan for beam (DELAY = 7), 
   * fine (1ns) TTC delay scan for beam (DELAY_TTC = 8), 
   * APV pulse shape tuning using peak mode operation (CALIBRATION = 3), 
   * APV pulse shape tuning using deconvolution mode operation (CALIBRATION_DECO = 33), 
   * APV pulse shape tuning with isha/vfs scan using peak mode operation (CALIBRATION = 3), 
   * APV pulse shape tuning with isha/vfs scan using deconvolution mode operation (CALIBRATION_DECO = 33), 
   * physics data-taking run (PHYSIC = 1), 
   * scope mode running (SCOPE_MODE = 15) 
   * multi mode operation (PHYSIC10 = 10), 
   */
  enum RunType { UNKNOWN_RUN_TYPE   = sistrip::unknown_,
		 UNDEFINED_RUN_TYPE = sistrip::invalid_,
		 FAST_CABLING          = 21,
		 FED_CABLING           = 13,
		 APV_TIMING            = 5,
		 FED_TIMING            = 12,
		 OPTO_SCAN             = 4,
		 VPSP_SCAN             = 14,
		 PEDESTALS             = 2,
		 APV_LATENCY           = 6,
                 FINE_DELAY            = 17,
		 DAQ_SCOPE_MODE        = 15,
		 CALIBRATION_SCAN      = 19,
		 CALIBRATION_SCAN_DECO = 20,
		 CALIBRATION           = 3,
		 CALIBRATION_DECO      = 33,
		 PHYSICS               = 1
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForRunType_H


