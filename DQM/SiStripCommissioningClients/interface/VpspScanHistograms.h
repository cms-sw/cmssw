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

};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


