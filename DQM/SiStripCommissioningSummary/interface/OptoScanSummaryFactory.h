#ifndef DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class OptoScanSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 protected:
  
  void extract( Iterator );
  
  void format();
  
};

#endif // DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
