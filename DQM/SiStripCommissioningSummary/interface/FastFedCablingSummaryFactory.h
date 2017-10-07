#ifndef DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class FastFedCablingSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator ) override;
  
  void format() override;
  
};

#endif // DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H
