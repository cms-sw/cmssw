#ifndef DQM_SiStripCommissioningAnalysis_FedTimingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_FedTimingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

/** 
   @class FedTimingAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for FED delay-FPGA "monitorables".
*/
class FedTimingAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  FedTimingAnalysis() {;}

  /** Destructor */

  virtual ~FedTimingAnalysis() {;}
  
  /** Takes a vector containing one TProfile of a tick mark and returns a vector of 2 unsigned shorts representing the rise time of the tick. The first is a coarse-time measurement (units of 25ns), the second a fine-time (units of 1ns). */
  virtual void analysis( const vector<const TProfile*>& histos, 
			      vector<unsigned short>& monitorables);
  
};

#endif // DQM_SiStripCommissioningAnalysis_FedTimingAnalysis_H

