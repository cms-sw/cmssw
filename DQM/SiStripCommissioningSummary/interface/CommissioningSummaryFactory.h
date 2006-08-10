#ifndef DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H

#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"

template<>
class SummaryHistogramFactory<ApvTimingAnalysis::Monitorables> {
  
 public:
  
  std::auto_ptr<TH1> summary( const sistrip::SummaryHisto&, 
			      const sistrip::SummaryType&, 
			      const sistrip::View&, 
			      const std::string& directory, 
			      const std::map<uint32_t,ApvTimingAnalysis::Monitorables>& );
  
};

#endif // DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
