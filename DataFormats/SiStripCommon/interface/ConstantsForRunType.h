// Last commit: $Id: ConstantsForRunType.h,v 1.7 2007/11/29 17:08:03 bainbrid Exp $

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
   * physics data-taking run                (1 = PHYSIC), 
   * FED calibration run for peds and noise (2 = PEDESTAL), 
   * pulse shape tuning using peak mode     (3 = CALIBRATION), 
   * bias and gain scan for LLD device      (4 = GAINSCAN),  
   * relative synch                         (5 = TIMING), 
   * coarse (25ns) latency scan for beam    (6 = LATENCY),
   * fine (1ns) PLL delay scan for beam     (7 = DELAY), 
   * fine (1ns) TTC delay scan for beam     (8 = DELAY_TTC), 
   * multi mode operation                   (10 = PHYSIC10), 
   * connection run                         (11 = CONNECTION),
   * relative APV synch using FED delays    (12 = TIMING_FED), 
   * connection run                         (13 = BARE_CONNECTION), 
   * baseline scan                          (14 = VPSPSCAN), 
   * scope mode running                     (15 = SCOPE) 
   * connection run                         (16 = FAST_CONNECTION),
   * fine delay at for layer                (17 = DELAY_LAYER) 
   * physics run in ZS mode                 (18 = PHYSIC_ZS) 
   * isha/vfs scan using peak mode          (19 = CALIBRATION_SCAN), 
   * isha/vfs scan using decon mode         (20 = CALIBRATION_SCAN_DECO), 
   * "fast" connection run                  (21 = XTOFS_CONNECTION), 
   * pulse shape tuning using decon mode    (33 = CALIBRATION_DECO), 
   */
  enum RunType { PHYSICS               = 1,
		 PEDESTALS             = 2,
		 CALIBRATION           = 3,
		 OPTO_SCAN             = 4,
		 APV_TIMING            = 5,
		 APV_LATENCY           = 6,
		 FINE_DELAY_PLL        = 7,
		 FINE_DELAY_TTC        = 8,
		 MULTI_MODE            = 10,
		 FED_TIMING            = 12,
                 FED_CABLING           = 13,
		 VPSP_SCAN             = 14,
		 DAQ_SCOPE_MODE        = 15,
                 QUITE_FAST_CABLING    = 16,
		 FINE_DELAY            = 17,
		 PHYSICS_ZS            = 18,
		 CALIBRATION_SCAN      = 19,
		 CALIBRATION_SCAN_DECO = 20,
		 FAST_CABLING          = 21,
		 CALIBRATION_DECO      = 33,
		 UNKNOWN_RUN_TYPE   = sistrip::unknown_,
		 UNDEFINED_RUN_TYPE = sistrip::invalid_
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForRunType_H


