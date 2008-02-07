#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class OptoScanHistograms : public virtual CommissioningHistograms {

 public:

  OptoScanHistograms( MonitorUserInterface* );
  OptoScanHistograms( DaqMonitorBEInterface* );
  virtual ~OptoScanHistograms();
  
  void histoAnalysis( bool debug );

  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistograms_H


