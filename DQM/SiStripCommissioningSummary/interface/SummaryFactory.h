#ifndef DQM_SiStripCommissioningSummary_SummaryFactory_H
#define DQM_SiStripCommissioningSummary_SummaryFactory_H

#include <string>

class CommissioningSummary;

class SummaryFactory {
  
 public:
  
  /** Defines the various summary histograms available. */
  enum Histo { UNKNOWN_HISTO, UNDEFINED_HISTO,
	       APV_TIMING_COARSE, APV_TIMING_FINE, APV_TIMING_DELAY, APV_TIMING_ERROR, APV_TIMING_BASE, APV_TIMING_PEAK, APV_TIMING_HEIGHT 
	       //@@ Other summary histo types here...
  };
  
  static CommissioningSummary* book( const Histo&, const std::string& directory );
  
};

#endif // DQM_SiStripCommissioningSummary_SummaryFactory_H


