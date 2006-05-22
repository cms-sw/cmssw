#ifndef DataFormats_SiStripCommon_SiStripEnumeratedTypes_H
#define DataFormats_SiStripCommon_SiStripEnumeratedTypes_H

#include "boost/cstdint.hpp"

namespace sistrip { 

  /** */
  enum FedReadoutMode { SCOPE_MODE = 0, 
			VIRGIN_RAW = 1, 
			PROC_RAW = 2, 
			ZERO_SUPPR = 3, 
			UNKNOWN_FED_MODE = 99 };


  /** Detector views. */
  enum View { UNKNOWN_VIEW, 
	      NO_VIEW, 
	      READOUT, 
	      CONTROL, 
	      DETECTOR };
  
  /** 
   * Commissioning Tasks (equivalent "TrackerSupervisor" enums in brackets): 
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
   * unknown commissioning task (UNKNOWN_TASK = 0),
   * no commissioning task (NO_TASK = 99),
   */
  enum Task { UNKNOWN_TASK =  0,   
	      NO_TASK      = 99, 
	      PHYSICS      =  1, 
	      FED_CABLING  = 13, 
	      APV_TIMING   =  5, 
	      FED_TIMING   = 12, 
	      OPTO_SCAN    =  4, 
	      VPSP_SCAN    = 14, 
	      PEDESTALS    =  2, 
	      APV_LATENCY  =  6 };
  
  /** 
   * Histogram contents:
   */
  enum Contents { UNKNOWN_CONTENTS, COMBINED, SUM2, SUM, NUM };

  /**
   * Key type used as an identifier:
   * 
   *
   *
   */
  enum KeyType { UNKNOWN_KEY, NO_KEY, FED, FEC, DET };

  
  enum Granularity { UNKNOWN_GRAN, MODULE, LLD_CHAN, APV_PAIR, APV };
  
}

#endif // DataFormats_SiStripCommon_SiStripEnumeratedTypes_H


