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
  
  typedef SummaryPlotFactory<ApvTimingAnalysis*> Factory;
  
  /** */
  void histoAnalysis( bool debug );
  
  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );
  
 protected: 
  
  std::map<uint32_t,ApvTimingAnalysis*> data_;
  
  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

