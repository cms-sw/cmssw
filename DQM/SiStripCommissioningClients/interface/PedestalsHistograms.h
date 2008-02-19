#ifndef DQM_SiStripCommissioningClients_PedestalsHistograms_H
#define DQM_SiStripCommissioningClients_PedestalsHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class PedestalsHistograms : public virtual CommissioningHistograms {

 public:
  
  PedestalsHistograms( MonitorUserInterface* );
  PedestalsHistograms( DaqMonitorBEInterface* );
  virtual ~PedestalsHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
