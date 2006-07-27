#ifndef DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include <string>

class CommissioningSummary;

class CommissioningSummaryFactory {
  
 public:
  
  /** Defines the various summary histograms available. */
  enum Histo { UNKNOWN_HISTO, UNDEFINED_HISTO,
	       APV_TIMING_COARSE, APV_TIMING_FINE, APV_TIMING_DELAY, APV_TIMING_ERROR, APV_TIMING_BASE, APV_TIMING_PEAK, APV_TIMING_HEIGHT 
	       //@@ Other summary histo types here...
  };
  
  static CommissioningSummary* book( const sistrip::Task&,
				     const Histo&, 
				     const std::string& directory );
  
};

#endif // DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H


