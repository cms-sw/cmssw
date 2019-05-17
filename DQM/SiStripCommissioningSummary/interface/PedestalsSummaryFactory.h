#ifndef DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
#define DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class PedestalsSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
protected:
  void extract(Iterator) override;

  void format() override;
};

#endif  // DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
