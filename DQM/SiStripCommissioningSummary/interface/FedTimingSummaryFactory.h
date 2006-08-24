#ifndef DQM_SiStripCommissioningSummary_FedTimingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_FedTimingSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedTimingAnalysis.h"

template<>
class SummaryHistogramFactory<FedTimingAnalysis::Monitorables> {
  
 public:
  
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,FedTimingAnalysis::Monitorables>& data,
		 TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_FedTimingSummaryFactory_H
