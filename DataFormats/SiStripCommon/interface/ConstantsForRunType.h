// Last commit: $Id: ConstantsForRunType.h,v 1.8 2008/01/14 09:17:15 bainbrid Exp $

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
  
  static const std::string unknownRunType_      = "UnknownRunType";
  static const std::string undefinedRunType_    = "UndefinedRunType";
  
  static const std::string fastCablingRun_      = "FastCabling";
  static const std::string fedCablingRun_       = "FedCabling";
  static const std::string apvTimingRun_        = "ApvTiming";
  static const std::string fedTimingRun_        = "FedTiming";
  static const std::string optoScanRun_         = "OptoScan";
  static const std::string vpspScanRun_         = "VpspScan";
  static const std::string pedestalsRun_        = "Pedestals";
  static const std::string pedsOnlyRun_         = "PedsOnly";
  static const std::string noiseRun_            = "Noise";
  static const std::string apvLatencyRun_       = "ApvLatency";
  static const std::string fineDelayRun_        = "FineDelay";
  static const std::string calibPeakRun_        = "CalibrationPeak";
  static const std::string calibDeconRun_       = "CalibrationDeco";
  static const std::string calibScanPeakRun_    = "CalibrationScanPeak";
  static const std::string calibScanDeconRun_   = "CalibrationScanDeco";
  static const std::string daqScopeModeRun_     = "DaqScopeMode";
  static const std::string physicsRun_          = "Physics";
  
  // ---------- Enumerated type ---------- 
  
  /** 
   * Run types: (equivalent "TrackerSupervisor" enums in brackets): 
   * unknown run type,
   * undefined run type,
   * physics data-taking run                  (1 = PHYSIC), 
   * FED calibration run for peds and noise   (2 = PEDS_AND_NOISE), 
   * pulse shape tuning using peak mode       (3 = CALIBRATION), 
   * bias and gain scan for LLD device        (4 = GAINSCAN),  
   * relative synch                           (5 = TIMING), 
   * coarse (25ns) latency scan for beam      (6 = LATENCY),
   * fine (1ns) PLL delay scan for beam       (7 = DELAY), 
   * fine (1ns) TTC delay scan for beam       (8 = DELAY_TTC), 
   * multi mode operation                     (10 = PHYSIC10), 
   * connection run                           (11 = CONNECTION),
   * relative APV synch using FED delays      (12 = TIMING_FED), 
   * connection run                           (13 = BARE_CONNECTION), 
   * baseline scan                            (14 = VPSPSCAN), 
   * scope mode running                       (15 = SCOPE) 
   * connection run                           (16 = FAST_CONNECTION),
   * fine delay at for layer                  (17 = DELAY_LAYER) 
   * physics run in ZS mode                   (18 = PHYSIC_ZS) 
   * isha/vfs scan using peak mode            (19 = CALIBRATION_SCAN), 
   * isha/vfs scan using decon mode           (20 = CALIBRATION_SCAN_DECO), 
   * "fast" connection run                    (21 = XTOFS_CONNECTION), 
   * FED calibration run for pedestals (only) (22 = PEDESTAL), 
   * FED calibration run for noise (only)     (23 = NOISE), 
   * pulse shape tuning using decon mode      (33 = CALIBRATION_DECO), 
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
		 PEDS_ONLY             = 22,
		 NOISE                 = 23,
		 CALIBRATION_DECO      = 33,
		 UNKNOWN_RUN_TYPE   = sistrip::unknown_,
		 UNDEFINED_RUN_TYPE = sistrip::invalid_
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForRunType_H


