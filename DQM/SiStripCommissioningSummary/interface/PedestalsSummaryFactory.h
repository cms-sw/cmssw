#ifndef DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
#define DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"

template<>
class SummaryHistogramFactory<PedestalsAnalysis::Monitorables> {
  
 public:
  
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,PedestalsAnalysis::Monitorables>& data,
		 TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
