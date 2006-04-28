#ifndef DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H
#define DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

/** 
    @file : DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h
    @class : PedestalsAnalysis
    @author : M. Wingham
    @brief : Histogram-based analysis for pedestals and noise "monitorables".
*/
class PedestalsAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  PedestalsAnalysis() {;}

  /** Destructor */
  virtual ~PedestalsAnalysis() {;}
  

  /** Takes a vector containing one TH1F of pedestals (error bars - raw noise) and one of residuals(error bars - noise) and fills the monitorables vector with 2 vectors - the first of pedestals, the second of noise. */
  virtual void analysis( const vector<const TProfile*>& histos, 
			      vector< vector<float> >& monitorables );
};

#endif // DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H

