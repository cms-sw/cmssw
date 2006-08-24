#ifndef DQM_SiStripCommissioningSummary_VpspScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_VpspScanSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/VpspScanAnalysis.h"

template<>
class SummaryHistogramFactory<VpspScanAnalysis::Monitorables> {
  
 public:
    
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,VpspScanAnalysis::Monitorables>& data,
		 TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_VpspScanSummaryFactory_H
