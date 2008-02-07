#ifndef DQM_SiStripCommissioningSummary_FineDelaySummaryFactory_H
#define DQM_SiStripCommissioningSummary_FineDelaySummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryHistogramFactory.h"
#include "CondFormats/SiStripObjects/interface/FineDelayAnalysis.h"

class SummaryGenerator;

template<>
class SummaryHistogramFactory<FineDelayAnalysis> {
  
 public:
  
  SummaryHistogramFactory();
  ~SummaryHistogramFactory();

  void init( const sistrip::Monitorable&, 
	     const sistrip::Presentation&,
	     const sistrip::View&, 
	     const std::string& top_level_dir, 
	     const sistrip::Granularity& );
  
  uint32_t extract( const std::map<uint32_t,FineDelayAnalysis>& data );
  
  void fill( TH1& summary_histo );
  
 private:
  
  sistrip::Monitorable mon_;
  sistrip::Presentation pres_;
  sistrip::View view_;
  std::string level_;
  sistrip::Granularity gran_;
  SummaryGenerator* generator_;
  
};

#endif // DQM_SiStripCommissioningSummary_FineDelaySummaryFactory_H
