

#ifndef DQM_SiStripCommissioningAnalysis_BiasGainAnalysis_H
#define DQM_SiStripCommissioningAnalysis_BiasGainAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

class BiasGainHistograms;
class BiasGainMonitorables;

/** 
   @class BiasGainAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for Opto- bias/gain "monitorables".
*/
class BiasGainAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  BiasGainAnalysis() {;}

  /** Destructor */
  virtual ~BiasGainAnalysis() {;}
  
  /** Takes a vector containing one TH1F of median tick height measurements and one of median tick base measurements vs. LLD bias. Both histograms correspond to a fixed gain setting. A vector of floating points is filled with two values: the first being the optimum bias setting for the LLD channel, and the second the gain measurement. */
  virtual void analysis( const vector<const TH1F*>& histos, 
			      vector<float>& monitorables );
  
};

#endif // DQM_SiStripCommissioningAnalysis_BiasGainAnalysis_H

