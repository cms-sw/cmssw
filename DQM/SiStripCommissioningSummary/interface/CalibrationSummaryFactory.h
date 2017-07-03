#ifndef DQM_SiStripCommissioningSummary_CalibrationSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CalibrationSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class CalibrationSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {

 protected:

   void extract( Iterator ) override;

   void format() override;

};
     
#endif // DQM_SiStripCommissioningSummary_CalibrationSummaryFactory_H
