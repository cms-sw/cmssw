#ifndef DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAnalysis.h"

template<>
class SummaryHistogramFactory<OptoScanAnalysis::Monitorables> {
  
 public:
  
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,OptoScanAnalysis::Monitorables>& data,
		 TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
