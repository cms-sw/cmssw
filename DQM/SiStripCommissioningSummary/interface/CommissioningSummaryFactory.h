#ifndef DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include <map>
#include <cstdint>

class CommissioningAnalysis;

template <>
class SummaryPlotFactory<CommissioningAnalysis*> : public SummaryPlotFactoryBase {
public:
  SummaryPlotFactory<CommissioningAnalysis*>() { ; }

  virtual ~SummaryPlotFactory<CommissioningAnalysis*>() { ; }

  typedef std::map<uint32_t, CommissioningAnalysis*>::const_iterator Iterator;

  uint32_t init(const sistrip::Monitorable&,
                const sistrip::Presentation&,
                const sistrip::View&,
                const std::string& top_level_dir,
                const sistrip::Granularity&,
                const std::map<uint32_t, CommissioningAnalysis*>& data);

  void fill(TH1& summary_histo);

protected:
  virtual void extract(Iterator) { ; }

  virtual void format() { ; }
};

#endif  // DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
