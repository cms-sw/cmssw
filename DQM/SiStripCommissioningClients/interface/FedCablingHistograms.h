#ifndef DQM_SiStripCommissioningClients_FedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class FedCablingHistograms : public CommissioningHistograms {

 public:
  
  /** */
  FedCablingHistograms( MonitorUserInterface* );
  /** */
  virtual ~FedCablingHistograms();
  
};

#endif // DQM_SiStripCommissioningClients_FedCablingHistograms_H


