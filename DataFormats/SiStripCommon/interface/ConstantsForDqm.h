// Last commit: $Id: ConstantsForDqm.h,v 1.13 2010/01/04 18:44:33 lowette Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForDqm_H
#define DataFormats_SiStripCommon_ConstantsForDqm_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/**
   @file ConstantsForDqm.h
   @brief Various generic constants used by DQM
*/

namespace sistrip { 
  
  // ---------- Useful characters ----------
  
  static const char dir_[]  = "/";
  static const char sep_[]  = "_";
  static const char pipe_[] = "|";
  static const char dot_[]  = ".";
  static const char hex_[]  = "0x";

  // ---------- Naming dirs and histos ----------

  static const char dqmSourceFileName_[] = "SiStripCommissioningSource";
  static const char dqmClientFileName_[] = "SiStripCommissioningClient";

  static const char dqmRoot_[]     = "DQMData";
  static const char collate_[]     = "Collate";
  static const char root_[]        = "SiStrip";
  static const char taskId_[]      = "SiStripCommissioningTask";
  static const char summaryPlot_[] = "SummaryPlot";
  static const char runNumber_[]   = "RunNumber";

  namespace extrainfo { 

    // ---------- opto scan ----------
    
    static const char gain_[]        = "Gain";
    static const char digital_[]     = "Digital";
    static const char baselineRms_[] = "BaselineNoise";

    // ---------- peds and noise ----------
    
    static const char pedestals_[]      = "Pedestals";
    static const char rawNoise_[]       = "RawNoise";
    static const char noise_[]          = "Noise";
    static const char commonMode_[]     = "CommonMode";
    static const char roughPedestals_[] = "RoughPedestals";
    static const char noiseProfile_[]   = "NoiseProfile";
    static const char noise2D_[]        = "Noise2D";
    
    static const char pedsAndRawNoise_[]   = "PedsAndRawNoise";          //@@ LEGACY
    static const char residualsAndNoise_[] = "ResidualsAndNoise";        //@@ LEGACY
    static const char pedsAndCmSubNoise_[] = "PedsAndCMSubtractedNoise"; //@@ LEGACY

    // ---------- latency ----------

    static const char clusterCharge_[]     = "ClusterCharge";
    static const char occupancy_[]         = "Occupancy";
    
  }
  
  // ---------- Actions to be taken by web client ----------

  enum Action { UNKNOWN_ACTION        = sistrip::unknown_, 
		UNDEFINED_ACTION      = sistrip::invalid_, 
		NO_ACTION             = 0, 
		ANALYZE_HISTOS        = 1,
		SAVE_HISTOS_TO_DISK   = 2,
		CREATE_SUMMARY_HISTOS = 3, 
		CREATE_TRACKER_MAP    = 4,
		UPLOAD_TO_DATABASE    = 5
  };
  
}

#endif // DataFormats_SiStripCommon_ConstantsForDqm_H


