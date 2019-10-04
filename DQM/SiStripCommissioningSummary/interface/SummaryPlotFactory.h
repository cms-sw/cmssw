#ifndef DQM_SiStripCommissioningSummary_SummaryPlotFactory_H
#define DQM_SiStripCommissioningSummary_SummaryPlotFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "TH1.h"
#include <string>
#include <map>
#include <cstdint>

template <class T>
class SummaryPlotFactory : public SummaryPlotFactoryBase {
public:
  uint32_t init(const sistrip::Monitorable&,
                const sistrip::Presentation&,
                const sistrip::View&,
                const std::string& top_level_dir,
                const sistrip::Granularity&,
                const std::map<uint32_t, T>& data);

  void fill(TH1& summary_histo);
};

#endif  // DQM_SiStripCommissioningSummary_SummaryPlotFactory_H
