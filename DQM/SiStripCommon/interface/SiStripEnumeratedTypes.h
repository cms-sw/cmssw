#ifndef DQM_SiStripCommon_SiStripEnumeratedTypes_H
#define DQM_SiStripCommon_SiStripEnumeratedTypes_H

namespace sistrip { 
  
  /** Detector views. */
  enum View { UNKNOWN_VIEW, 
	      NO_VIEW, 
	      READOUT, 
	      CONTROL, 
	      DETECTOR 
  };
  
  /** Histogram contents. */
  enum Contents { UNKNOWN_CONTENTS, 
		  COMBINED, 
		  SUM2, 
		  SUM, 
		  NUM 
  };

  /** Key type used as an identifier. */
  enum KeyType { UNKNOWN_KEY, 
		 NO_KEY, 
		 FED, 
		 FEC, 
		 DET 
  };
  
  /** Histogram granularity. */
  enum Granularity { UNKNOWN_GRAN, 
		     MODULE, 
		     LLD_CHAN, 
		     APV_PAIR, 
		     APV 
  };
  
  /** Defines action to be taken by web client. */
  enum Action { UNKNOWN_ACTION = 9999, 
		NO_ACTION = 0, 
		ANALYZE_HISTOS = 1,
		SAVE_HISTOS_TO_DISK = 2,
		CREATE_SUMMARY_HISTOS = 3, 
		CREATE_TRACKER_MAP = 4,
		UPLOAD_TO_DATABASE = 5
  };

  /** Defines the type of summary histogram. */
  enum SummaryType { UNKNOWN_SUMMARY_TYPE = 9999, 
		     UNDEFINED_SUMMARY_TYPE = 0, 
		     SUMMARY_DISTR = 1,
		     SUMMARY_1D = 2,
		     SUMMARY_2D = 3
  };
 
  /** Defines the various summary histograms available. */
  enum SummaryHisto { UNKNOWN_SUMMARY_HISTO = 9999, 
		      UNDEFINED_SUMMARY_HISTO = 0, 
		      // FED CABLING
		      FED_CABLING_FED_ID = 1301, 
		      FED_CABLING_FED_CH = 1301, 
		      // APV TIMING
		      APV_TIMING_PLL_COARSE = 501, 
		      APV_TIMING_PLL_FINE = 502, 
		      APV_TIMING_DELAY = 503, 
		      APV_TIMING_ERROR = 504, 
		      APV_TIMING_BASE = 505, 
		      APV_TIMING_PEAK = 506, 
		      APV_TIMING_HEIGHT = 507,
		      // FED TIMING
		      FED_TIMING_PLL_COARSE = 1201, 
		      FED_TIMING_PLL_FINE = 1202, 
		      FED_TIMING_DELAY = 1203, 
		      FED_TIMING_ERROR = 1204, 
		      FED_TIMING_BASE = 1205, 
		      FED_TIMING_PEAK = 1206, 
		      FED_TIMING_HEIGHT = 1207,
		      // OPTO SCAN
		      OPTO_SCAN_LLD_GAIN_SETTING = 401,
		      OPTO_SCAN_LLD_BIAS_SETTING = 402,
		      OPTO_SCAN_MEASURED_GAIN = 403, 
		      OPTO_SCAN_ZERO_LIGHT_LEVEL = 404, 
		      OPTO_SCAN_LINK_NOISE = 405,
		      OPTO_SCAN_BASELINE_LIFT_OFF = 406,
		      OPTO_SCAN_LASER_THRESHOLD = 407,  
		      OPTO_SCAN_TICK_HEIGHT = 408,
		      // VPSP SCAN
		      VPSP_SCAN_APV0 = 1401, 
		      VPSP_SCAN_APV1 = 1402, 
		      // PEDESTALS / NOISE
		      PEDESTALS_MEAN = 201, 
		      PEDESTALS_SPREAD = 202, 
		      PEDESTALS_MAX = 203, 
		      PEDESTALS_MIN = 204, 
		      NOISE_MEAN = 205, 
		      NOISE_SPREAD = 206, 
		      NOISE_MAX = 207, 
		      NOISE_MIN = 208, 
		      NUM_OF_DEAD = 209, 
		      NUM_OF_NOISY = 210
		      //@@ add other summary histos here... 
  };
  
}

#endif // DQM_SiStripCommon_SiStripEnumeratedTypes_H


