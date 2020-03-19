#ifndef DQM_SiStripCommissioningSummary_CalibrationScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CalibrationScanSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class CalibrationScanSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
protected:
  void extract(Iterator) override;

  void format() override;
};

#endif  // DQM_SiStripCommissioningSummary_CalibrationScanSummaryFactory_H
