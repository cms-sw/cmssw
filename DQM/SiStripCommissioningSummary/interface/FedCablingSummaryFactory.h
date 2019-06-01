#ifndef DQM_SiStripCommissioningSummary_FedCablingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_FedCablingSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "CondFormats/SiStripObjects/interface/FedCablingAnalysis.h"

template <>
class SummaryPlotFactory<FedCablingAnalysis*> : public SummaryPlotFactoryBase {
public:
  uint32_t init(const sistrip::Monitorable&,
                const sistrip::Presentation&,
                const sistrip::View&,
                const std::string& top_level_dir,
                const sistrip::Granularity&,
                const std::map<uint32_t, FedCablingAnalysis*>& data);

  void fill(TH1& summary_histo);
};

#endif  // DQM_SiStripCommissioningSummary_FedCablingSummaryFactory_H
