// Last commit: $Id: ConstantsForRunType.h,v 1.11 2009/10/22 15:57:40 lowette Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForRunType_H
#define DataFormats_SiStripCommon_ConstantsForRunType_H

#include "DataFormats/SiStripCommon/interface/Constants.h"

/** 
    @file ConstantsForRunType.h

    @brief Constants and enumerated type for sistrip::RunType
*/

namespace sistrip { 
  
  // ---------- Constants ---------- 
  
  static const char unknownRunType_[]       = "UnknownRunType";
  static const char undefinedRunType_[]     = "UndefinedRunType";
  
  static const char fastCablingRun_[]       = "FastCabling";
  static const char fedCablingRun_[]        = "FedCabling";
  static const char apvTimingRun_[]         = "ApvTiming";
  static const char fedTimingRun_[]         = "FedTiming";
  static const char optoScanRun_[]          = "OptoScan";
  static const char vpspScanRun_[]          = "VpspScan";
  static const char pedestalsRun_[]         = "Pedestals";
  static const char pedsOnlyRun_[]          = "PedsOnly";
  static const char noiseRun_[]             = "Noise";
  static const char pedsFullNoiseRun_[]     = "PedsFullNoise";
  static const char apvLatencyRun_[]        = "ApvLatency";
  static const char fineDelayRun_[]         = "FineDelay";
  static const char calibPeakRun_[]         = "CalibrationPeak";
  static const char calibDeconRun_[]        = "CalibrationDeco";
  static const char calibScanPeakRun_[]     = "CalibrationScanPeak";
  static const char calibScanDeconRun_[]    = "CalibrationScanDeco";
  static const char daqScopeModeRun_[]      = "DaqScopeMode";
  static const char physicsRun_[]           = "Physics";
  
  // ---------- Enumerated type ---------- 
  
  /** 
   * Run types: (equivalent "TrackerSupervisor" enums in brackets): 
   * unknown run type,
   * undefined run type,
   * physics data-taking run                  (1 = PHYSICS), 
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
   * FED calib. for peds and detailed noise   (24 = PEDS_FULL_NOISE),
   * pulse shape tuning using decon mode      (33 = CALIBRATION_DECO), 
   */
  enum RunType { 
      PHYSICS               = 1,
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
      PEDS_FULL_NOISE       = 24,
      CALIBRATION_DECO      = 33,
      UNKNOWN_RUN_TYPE   = sistrip::unknown_,
      UNDEFINED_RUN_TYPE = sistrip::invalid_
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForRunType_H


