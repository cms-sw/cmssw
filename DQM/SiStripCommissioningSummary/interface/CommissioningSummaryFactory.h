#ifndef DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"

template<>
class SummaryHistogramFactory<ApvTimingAnalysis::Monitorables> {
  
 public:
  
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,ApvTimingAnalysis::Monitorables>& data,
		 TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
