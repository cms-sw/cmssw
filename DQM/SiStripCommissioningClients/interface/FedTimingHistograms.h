#ifndef DQM_SiStripCommissioningClients_FedTimingHistograms_H
#define DQM_SiStripCommissioningClients_FedTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class FedTimingHistograms : public CommissioningHistograms {

 public:
  
  /** */
  FedTimingHistograms( MonitorUserInterface* );
  /** */
  virtual ~FedTimingHistograms();
  
};

#endif // DQM_SiStripCommissioningClients_FedTimingHistograms_H


