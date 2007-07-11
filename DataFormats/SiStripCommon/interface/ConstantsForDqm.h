// Last commit: $Id: ConstantsForDqm.h,v 1.4 2007/06/04 12:47:22 bainbrid Exp $

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
  
  static const std::string dir_  = "/";
  static const std::string sep_  = "_";
  static const std::string pipe_ = "|";
  static const std::string dot_  = ".";
  static const std::string hex_  = "0x";

  // ---------- Naming dirs and histos ----------

  static const std::string dqmSourceFileName_ = "SiStripCommissioningSource";
  static const std::string dqmClientFileName_ = "SiStripCommissioningClient";

  static const std::string dqmRoot_     = "DQMData";
  static const std::string root_        = "SiStrip";
  static const std::string taskId_      = "SiStripCommissioningTask";
  static const std::string summaryPlot_ = "SummaryPlot";
  static const std::string runNumber_   = "RunNumber";

  static const std::string gain_              = "Gain";
  static const std::string digital_           = "Digital";
  static const std::string baselineRms_       = "BaselineNoise";
  static const std::string pedsAndRawNoise_   = "PedsAndRawNoise";
  static const std::string residualsAndNoise_ = "ResidualsAndNoise";
  static const std::string commonMode_        = "CommonMode";

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


