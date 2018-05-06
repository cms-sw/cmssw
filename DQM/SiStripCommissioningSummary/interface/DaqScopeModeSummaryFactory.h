#ifndef DQM_SiStripCommissioningSummary_DaqScopeModeSummaryFactory_H
#define DQM_SiStripCommissioningSummary_DaqScopeModeSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class DaqScopeModeSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator );
  
  void format();
  
};

#endif // DQM_SiStripCommissioningSummary_DaqScopeModeSummaryFactory_H
