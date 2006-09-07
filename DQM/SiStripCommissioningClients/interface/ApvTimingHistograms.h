#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/ApvTimingSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"

class MonitorUserInterface;

class ApvTimingHistograms : public CommissioningHistograms {

 public:
  
  ApvTimingHistograms( MonitorUserInterface* );
  virtual ~ApvTimingHistograms();

  typedef SummaryHistogramFactory<ApvTimingAnalysis> Factory;
  
  /** */
  void histoAnalysis();

  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&,
			   const sistrip::SummaryType&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );

 protected: 
  
  std::map<uint32_t,ApvTimingAnalysis> data_;

  std::auto_ptr<Factory> factory_;
  
  const float optimumSamplingPoint_;
  float minDelay_;
  float maxDelay_; 
  uint32_t deviceWithMinDelay_;
  uint32_t deviceWithMaxDelay_;
 
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

