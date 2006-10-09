#ifndef DQM_SiStripCommissioningClients_PedestalsHistograms_H
#define DQM_SiStripCommissioningClients_PedestalsHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class PedestalsHistograms : public CommissioningHistograms {

 public:
  
  /** */
  PedestalsHistograms( MonitorUserInterface* );
  /** */
  virtual ~PedestalsHistograms();
  
};

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
