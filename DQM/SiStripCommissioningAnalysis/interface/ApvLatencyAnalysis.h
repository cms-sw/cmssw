#ifndef DQM_SiStripCommissioningAnalysis_ApvLatencyAnalysis_H
#define DQM_SiStripCommissioningAnalysis_ApvLatencyAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

/** 
   \class ApvLatencyAnalysis
   \brief Concrete implementation of the histogram-based analysis for
   ApvLatency "monitorables". 
*/
class ApvLatencyAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  ApvLatencyAnalysis() {;}

  /** Destructor */
  virtual ~ApvLatencyAnalysis() {;}
  
  /** Takes a vector containing one TH1F of a scan through coarse (25ns) trigger latency settings, filled with the number of recorded hits per setting. The monitorables vector is filled with the latency for the largest number of recorded hits over 5*sigma of the noise. */
 virtual void analysis( const vector<const TH1F*>& histos, 
			      vector<unsigned short>& monitorables );

};

#endif // DQM_SiStripCommissioningAnalysis_ApvLatencyAnalysis_H

