#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class VpspScanHistograms : public CommissioningHistograms {

 public:
  
  /** */
  VpspScanHistograms( MonitorUserInterface* );
  /** */
  virtual ~VpspScanHistograms();
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


