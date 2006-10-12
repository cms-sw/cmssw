#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include <vector>

class MonitorUserInterface;

class OptoScanHistograms : public CommissioningHistograms {

 public:

  /** */
  OptoScanHistograms( MonitorUserInterface* );
  /** */
  virtual ~OptoScanHistograms();
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistograms_H


