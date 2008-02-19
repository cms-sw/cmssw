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
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

