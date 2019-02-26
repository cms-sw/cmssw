#ifndef DQM_SiStripCommissioningClients_FedTimingHistograms_H
#define DQM_SiStripCommissioningClients_FedTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FedTimingSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/FedTimingAnalysis.h"


class FedTimingHistograms : public CommissioningHistograms {

 public:
  
  FedTimingHistograms( const edm::ParameterSet& pset, DQMStore* );
  ~FedTimingHistograms() override;

  typedef SummaryHistogramFactory<FedTimingAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug ) override;

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& ) override;

 protected:

  std::map<uint32_t,FedTimingAnalysis> data_;

  std::unique_ptr<Factory> factory_;
  
  const float optimumSamplingPoint_;
  float minDelay_;
  float maxDelay_; 
  uint32_t deviceWithMinDelay_;
  uint32_t deviceWithMaxDelay_;

};

#endif // DQM_SiStripCommissioningClients_FedTimingHistograms_H

