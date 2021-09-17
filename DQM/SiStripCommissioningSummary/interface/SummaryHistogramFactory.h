#ifndef DQM_SiStripCommissioningSummary_SummaryHistogramFactory_H
#define DQM_SiStripCommissioningSummary_SummaryHistogramFactory_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "TH1.h"
#include <string>
#include <map>
#include <cstdint>

class SummaryGenerator;

template <class T>
class SummaryHistogramFactory {
public:
  SummaryHistogramFactory();
  ~SummaryHistogramFactory();

  void init(const sistrip::Monitorable&,
            const sistrip::Presentation&,
            const sistrip::View&,
            const std::string& top_level_dir,
            const sistrip::Granularity&);

  uint32_t extract(const std::map<uint32_t, T>& data);

  void fill(TH1& summary_histo);

private:
  sistrip::Monitorable mon_;
  sistrip::Presentation pres_;
  sistrip::View view_;
  std::string level_;
  sistrip::Granularity gran_;
  SummaryGenerator* generator_;
};

#endif  // DQM_SiStripCommissioningSummary_SummaryHistogramFactory_H
