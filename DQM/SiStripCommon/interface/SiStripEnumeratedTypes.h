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
 
  /** Defines the various summary histograms available. */
  enum SummaryHisto { UNKNOWN_SUMMARY_HISTO, 
		      UNDEFINED_SUMMARY_HISTO, 
		      APV_TIMING_COARSE, 
		      APV_TIMING_FINE, 
		      APV_TIMING_DELAY, 
		      APV_TIMING_ERROR, 
		      APV_TIMING_BASE, 
		      APV_TIMING_PEAK, 
		      APV_TIMING_HEIGHT 
		      //@@ add other summary histos here... 
  };
  
  /** Defines the type of summary histogram. */
  enum SummaryType { UNKNOWN_SUMMARY_TYPE, 
		     UNDEFINED_SUMMARY_TYPE, 
		     SUMMARY_SIMPLE_DISTR,
		     SUMMARY_LOGICAL_VIEW
  };
  
  /** Defines action to be taken by web client. */
  enum Action { UNKNOWN_ACTION, 
		NO_ACTION, 
		ANALYZE_HISTOS,
		SAVE_HISTOS_TO_DISK,
		CREATE_SUMMARY_HISTOS, 
		CREATE_TRACKER_MAP,
		UPLOAD_TO_DATABASE
  };
  
}

#endif // DQM_SiStripCommon_SiStripEnumeratedTypes_H


