#ifndef DQM_SiStripCommissioningSummary_ApvTimingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_ApvTimingSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class ApvTimingSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator );
  
  void format();
  
};

#endif // DQM_SiStripCommissioningSummary_ApvTimingSummaryFactory_H
