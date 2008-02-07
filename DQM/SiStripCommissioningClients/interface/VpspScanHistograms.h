#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class VpspScanHistograms : public virtual CommissioningHistograms {

 public:
  
  VpspScanHistograms( MonitorUserInterface* );
  VpspScanHistograms( DaqMonitorBEInterface* );
  virtual ~VpspScanHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );

};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


