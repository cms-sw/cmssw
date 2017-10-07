#ifndef DQM_SiStripCommissioningSummary_VpspScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_VpspScanSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class VpspScanSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator ) override;
  
  void format() override;
  
};

#endif // DQM_SiStripCommissioningSummary_VpspScanSummaryFactory_H
