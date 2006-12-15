#ifndef DataFormats_SiStripCommon_SiStripEnumeratedTypes_H
#define DataFormats_SiStripCommon_SiStripEnumeratedTypes_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

// -------------------- Commissioning tasks --------------------

namespace sistrip { 

  /** 
   * Commissioning Tasks (equivalent "TrackerSupervisor" enums in brackets): 
   * unknown commissioning task,
   * undefined commissioning task,
   * physics run (PHYSIC = 1), 
   * calibration run (PEDESTAL = 2), 
   * pulse shape tuning (CALIBRATION = 3), 
   * pulse shape tuning (CALIBRATION_DECO = 33), 
   * laser driver bias and gain (GAINSCAN = 4), 
   * relative apv synchronisation (TIMING = 5), 
   * coarse (25ns) apv latency scan for beam (LATENCY = 6),
   * fine (1ns) pll scan for beam (DELAY = 7), 
   * multi mode operation (PHYSIC10 = 10), 
   * connection of apv pairs to fed channels (CONNECTION = 11), 
   * fine (1ns) ttc scan for beam (DELAY_TTC = 8), 
   * relative apv synchronisation using fed delays (TIMING_FED = 12), 
   * connection of apv pairs to fed channels (BARE_CONNECTION = 13), 
   * apv baseline scan (VPSPSCAN = 14), 
   * DAQ scope mode readout (SCOPE = 15), 
   */
  enum Task { UNKNOWN_TASK = sistrip::unknown_,
	      UNDEFINED_TASK = sistrip::invalid_,
	      FED_CABLING = 13,
	      APV_TIMING = 5,
	      FED_TIMING = 12,
	      OPTO_SCAN = 4,
	      VPSP_SCAN = 14,
	      PEDESTALS = 2,
	      APV_LATENCY = 6,
	      DAQ_SCOPE_MODE = 15,
	      PHYSICS = 1
  };

}

// -------------------- "Sources" of cabling object info --------------------

namespace sistrip { 

  enum CablingSource { UNKNOWN_CABLING_SOURCE = sistrip::unknown_,
		       UNDEFINED_CABLING_SOURCE = sistrip::invalid_,
		       CABLING_FROM_CONNS = 1,
		       CABLING_FROM_DEVICES = 2,
		       CABLING_FROM_DETIDS = 3
  };
  
}

// -------------------- FED-related enumerated types --------------------

namespace sistrip { 

  /** */
  enum FedReadoutMode { UNKNOWN_FED_READOUT_MODE = sistrip::unknown_,
			UNDEFINED_FED_READOUT_MODE = sistrip::invalid_,
			SCOPE_MODE = 1, 
			VIRGIN_RAW = 2, 
			PROC_RAW = 6, 
			ZERO_SUPPR = 10,
			ZERO_SUPPR_LITE = 12
  };

  /** */
  enum FedReadoutPath { UNKNOWN_FED_READOUT_PATH = sistrip::unknown_,
			UNDEFINED_FED_READOUT_PATH = sistrip::invalid_,
			VME_READOUT = 1, 
			SLINK_READOUT = 2 
  };
  
  /** */
  enum FedBufferFormat { UNKNOWN_FED_BUFFER_FORMAT = sistrip::unknown_,
			 UNDEFINED_FED_BUFFER_FORMAT = sistrip::invalid_,
			 FULL_DEBUG_FORMAT = 1, 
			 APV_ERROR_FORMAT = 2 
  };
  
  /** */
  enum FedSuperMode { UNKNOWN_FED_SUPER_MODE = sistrip::unknown_,
		      UNDEFINED_FED_SUPER_MODE = sistrip::invalid_,
		      REAL = 0, 
		      FAKE = 1 
  };
  
}

// -------------------- DQM histograms --------------------

namespace sistrip { 
  
  /** Detector views. */
  enum View { UNKNOWN_VIEW = sistrip::unknown_, 
	      UNDEFINED_VIEW = sistrip::invalid_, 
	      READOUT = 1, 
	      CONTROL = 2, 
	      DETECTOR = 3 
  };
  
  /** Key type used as an identifier. */
  enum KeyType { UNKNOWN_KEY = sistrip::unknown_,  
		 UNDEFINED_KEY = sistrip::invalid_,  
		 //NO_KEY = 0, 
		 FED_KEY = 1, 
		 FEC_KEY = 2, 
		 DET_KEY = 3 
  };
  
  /** Histogram granularity. */
  enum Granularity { UNKNOWN_GRAN = sistrip::unknown_, 
		     UNDEFINED_GRAN = sistrip::invalid_, 
		     //NO_GRAN = 0,
		     TRACKER = 1, PARTITION = 2, TIB = 3, TOB = 4, TEC = 5,                  // System
		     FEC_CRATE = 6, FEC_SLOT = 7, FEC_RING = 8, CCU_ADDR = 9, CCU_CHAN = 10, // Control
		     FED = 11, FED_CHANNEL = 12, FE_UNIT = 13, FE_CHAN = 14,                 // Readout 
		     LAYER = 15, ROD = 16, STRING = 17, DISK = 18, PETAL = 19, RING = 20,    // Sub-structures 
		     MODULE = 21, LLD_CHAN = 22, APV = 23                                    // Module and below
  };
  
  /** Defines action to be taken by web client. */
  enum Action { UNKNOWN_ACTION = sistrip::unknown_, 
		UNDEFINED_ACTION = sistrip::invalid_, 
		NO_ACTION = 0, 
		ANALYZE_HISTOS = 1,
		SAVE_HISTOS_TO_DISK = 2,
		CREATE_SUMMARY_HISTOS = 3, 
		CREATE_TRACKER_MAP = 4,
		UPLOAD_TO_DATABASE = 5
  };
  
}

// -------------------- Summary plots --------------------

namespace sistrip { 
  
  /** Defines the presentation type for the summary histogram. */
  enum Presentation { UNKNOWN_PRESENTATION = sistrip::unknown_, 
		      UNDEFINED_PRESENTATION = sistrip::invalid_, 
		      SUMMARY_HISTO = 1,
		      SUMMARY_1D = 2,
		      SUMMARY_2D = 3,
		      SUMMARY_PROF = 4
  };
  
  /** Defines the monitorable for the summary histogram. */
  enum Monitorable { UNKNOWN_MONITORABLE = sistrip::unknown_, 
		     UNDEFINED_MONITORABLE = sistrip::invalid_, 
		     // FED CABLING
		     FED_CABLING_FED_ID = 1301, 
		     FED_CABLING_FED_CH = 1302, 
		     FED_CABLING_ADC_LEVEL = 1303, 
		     // APV TIMING
		     APV_TIMING_TIME = 501, 
		     APV_TIMING_MAX_TIME = 502, 
		     APV_TIMING_DELAY = 503, 
		     APV_TIMING_ERROR = 504, 
		     APV_TIMING_BASE = 505, 
		     APV_TIMING_PEAK = 506, 
		     APV_TIMING_HEIGHT = 507,
		     // FED TIMING
		     FED_TIMING_TIME = 501, 
		     FED_TIMING_MAX_TIME = 502, 
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
		     VPSP_SCAN_BOTH_APVS = 1401, 
		     VPSP_SCAN_APV0 = 1402, 
		     VPSP_SCAN_APV1 = 1403, 
		     // PEDESTALS / NOISE
		     PEDESTALS_ALL_STRIPS = 201, 
		     PEDESTALS_MEAN = 202, 
		     PEDESTALS_SPREAD = 203, 
		     PEDESTALS_MAX = 204, 
		     PEDESTALS_MIN = 205, 
		     NOISE_ALL_STRIPS = 206, 
		     NOISE_MEAN = 207, 
		     NOISE_SPREAD = 208, 
		     NOISE_MAX = 209, 
		     NOISE_MIN = 210, 
		     NUM_OF_DEAD = 211, 
		     NUM_OF_NOISY = 212,
		     // DAQ SCOPE MODE
		     DAQ_SCOPE_MODE_MEAN_SIGNAL = 1501
		     //@@ add other summary histos here... 
  };
  
}

#endif // DataFormats_SiStripCommon_SiStripEnumeratedTypes_H


