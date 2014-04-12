#ifndef DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
#define DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class PedestalsSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator );
  
  void format();
  
};

#endif // DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
