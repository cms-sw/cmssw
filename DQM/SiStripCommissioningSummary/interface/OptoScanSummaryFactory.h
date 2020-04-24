#ifndef DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class OptoScanSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator ) override;
  
  void format() override;
  
};

#endif // DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
