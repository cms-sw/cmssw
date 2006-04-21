#ifndef DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

/**
   @class : ApvTimingAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for PLL-skew "monitorables". 
*/

class ApvTimingAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  ApvTimingAnalysis() {;}

  /** Destructor */
  virtual ~ApvTimingAnalysis() {;}
  
  /** Takes a vector containing one TH1F of a tick mark and fills a vector of 2 unsigned shorts representing the rise time of the tick. The first is a coarse-time measurement (units of 25ns), the second a fine-time (units 1/24th of the coarse). */
  virtual void analysis( const vector<const TH1F*>& histos, 
			      vector<unsigned short>& monitorables);
};



#endif // DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

