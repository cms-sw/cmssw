#ifndef DQM_SiStripCommissioningSummary_FedCablingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_FedCablingSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAnalysis.h"

template<>
class SummaryHistogramFactory<FedCablingAnalysis::Monitorables> {
  
 public:
  
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,FedCablingAnalysis::Monitorables>& data,
		 TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_FedCablingSummaryFactory_H
