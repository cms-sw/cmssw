#ifndef DQM_SiStripCommissioningClients_FineDelayHistograms_H
#define DQM_SiStripCommissioningClients_FineDelayHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FineDelaySummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/FineDelayAnalysis.h"

class DQMOldReceiver;

class FineDelayHistograms : virtual public CommissioningHistograms {

 public:
  
  FineDelayHistograms( DQMStore* );
  FineDelayHistograms( DQMOldReceiver* );
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

