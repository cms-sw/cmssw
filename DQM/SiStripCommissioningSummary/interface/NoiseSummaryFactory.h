#ifndef DQM_SiStripCommissioningSummary_NoiseSummaryFactory_H
#define DQM_SiStripCommissioningSummary_NoiseSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class NoiseSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator ) override;
  
  void format() override;
  
};

#endif // DQM_SiStripCommissioningSummary_NoiseSummaryFactory_H
