#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class ApvTimingHistograms : public CommissioningHistograms {

 public:
  
  /** */
  ApvTimingHistograms( MonitorUserInterface* );
  /** */
  virtual ~ApvTimingHistograms();
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H


