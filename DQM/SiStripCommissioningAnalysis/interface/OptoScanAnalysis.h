#ifndef DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H
#define DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

/** 
   @class OptoScanAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for Opto- bias/gain "monitorables".
*/
class OptoScanAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  OptoScanAnalysis() {;}

  /** Destructor */
  virtual ~OptoScanAnalysis() {;}
  
  /** Takes a vector containing one TH1F of median tick height measurements and one of median tick base measurements vs. LLD bias. Both histograms correspond to a fixed gain setting. A vector of floating points is filled with two values: the first being the optimum bias setting for the LLD channel, and the second the gain measurement. */
  virtual void analysis( const vector<const TProfile*>& histos, 
			      vector<float>& monitorables );
  
};

#endif // DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

