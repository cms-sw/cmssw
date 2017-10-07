#ifndef DQM_SiStripCommissioningSummary_PedsFullNoiseSummaryFactory_H
#define DQM_SiStripCommissioningSummary_PedsFullNoiseSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class PedsFullNoiseSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator ) override;
  
  void format() override;
  
};

#endif // DQM_SiStripCommissioningSummary_PedsFullNoiseSummaryFactory_H
