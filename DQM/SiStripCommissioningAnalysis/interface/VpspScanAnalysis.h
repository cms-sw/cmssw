#ifndef DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H
#define DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

/** 
   @class : VpspScanAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for VPSP "monitorables". 
*/
class VpspScanAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  VpspScanAnalysis() {;}

  /** Destructor */
  virtual ~VpspScanAnalysis() {;}
  
  /** Takes a vector containing one TProfile of APV baseline measurements vs. VPSP setting and fills the monitorables vector with the optimum VPSP setting for this device. */
  virtual void analysis(const vector<const TProfile*>& histos, 
			      vector<unsigned short>& monitorables);
  
};

#endif // DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

