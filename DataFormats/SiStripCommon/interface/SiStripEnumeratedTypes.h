#ifndef DataFormats_SiStripCommon_SiStripEnumeratedTypes_H
#define DataFormats_SiStripCommon_SiStripEnumeratedTypes_H

#include "boost/cstdint.hpp"
#include <vector>

namespace sistrip { 

  /** */
  enum FedReadoutMode { UNKNOWN_FED_READOUT_MODE,
			UNDEFINED_FED_READOUT_MODE,
			SCOPE_MODE, 
			VIRGIN_RAW, 
			PROC_RAW, 
			ZERO_SUPPR,
			ZERO_SUPPR_LITE };

  /** */
  enum FedReadoutPath { UNKNOWN_FED_READOUT_PATH,
			UNDEFINED_FED_READOUT_PATH,
			VME_READOUT, 
			SLINK_READOUT };
  
  /** */
  enum FedBufferFormat { UNKNOWN_FED_BUFFER_FORMAT,
			 UNDEFINED_FED_BUFFER_FORMAT,
			 FULL_DEBUG_FORMAT, 
			 APV_ERROR_FORMAT };
  
  /** */
  enum FedSuperMode { UNKNOWN_FED_SUPER_MODE,
		      UNDEFINED_FED_SUPER_MODE,
		      REAL, 
		      FAKE };
  
  /** Detector views. */
  enum View { UNKNOWN_VIEW, 
	      NO_VIEW, 
	      READOUT, 
	      CONTROL, 
	      DETECTOR };
  
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
   * scope mode readout (SCOPE = 15), 
   */
  enum Task { UNKNOWN_TASK,
	      UNDEFINED_TASK,
	      FED_CABLING,
	      APV_TIMING,
	      FED_TIMING,
	      OPTO_SCAN,
	      VPSP_SCAN,
	      PEDESTALS,
	      APV_LATENCY,
	      PHYSICS };
  
  /** 
   * Histogram contents:
   */
  enum Contents { UNKNOWN_CONTENTS, COMBINED, SUM2, SUM, NUM };

  /**
   * Key type used as an identifier:
   */
  enum KeyType { UNKNOWN_KEY, NO_KEY, FED, FEC, DET };
  
  /**
   * Histogram granularity
   */
  enum Granularity { UNKNOWN_GRAN, MODULE, LLD_CHAN, APV_PAIR, APV };

}

#endif // DataFormats_SiStripCommon_SiStripEnumeratedTypes_H


