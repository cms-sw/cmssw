#ifndef DQM_SiStripCommissioningAnalysis_VPSPAnalysis_H
#define DQM_SiStripCommissioningAnalysis_VPSPAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <vector>

class VPSPHistograms;
class VPSPMonitorables;

/** 
   @class : VPSPAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for VPSP "monitorables". 
*/
class VPSPAnalysis : public CommissioningAnalysis {
  
 public:

  /** Constructor */
  VPSPAnalysis() {;}

  /** Destructor */
  virtual ~VPSPAnalysis() {;}
  
  /** Takes a vector containing one TH1F of APV baseline measurements vs. VPSP setting and fills the monitorables vector with the optimum VPSP setting for this device. */
  virtual void analysis(const vector<const TH1F*>& histos, 
			      vector<unsigned short>& monitorables);
  
};

#endif // DQM_SiStripCommissioningAnalysis_VPSPAnalysis_H

