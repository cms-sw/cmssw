#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class ApvTimingHistograms : public virtual CommissioningHistograms {

 public:
  
  ApvTimingHistograms( MonitorUserInterface* );
  ApvTimingHistograms( DaqMonitorBEInterface* );
  virtual ~ApvTimingHistograms();
  
  void histoAnalysis( bool debug );

  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

