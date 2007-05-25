#ifndef DQM_SiStripCommissioningClients_FineDelayHistograms_H
#define DQM_SiStripCommissioningClients_FineDelayHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FineDelaySummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FineDelayAnalysis.h"

class MonitorUserInterface;

class FineDelayHistograms : public CommissioningHistograms {

 public:
  
  FineDelayHistograms( MonitorUserInterface* );
  virtual ~FineDelayHistograms();
  
  typedef SummaryHistogramFactory<FineDelayAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
                           const sistrip::Presentation&,
                           const std::string& top_level_dir,
                           const sistrip::Granularity& );
  
 protected: 
  
  std::map<uint32_t,FineDelayAnalysis> data_;
  
  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistograms_H

