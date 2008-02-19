#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class FastFedCablingHistograms : public virtual CommissioningHistograms {

 public:
  
  FastFedCablingHistograms( MonitorUserInterface* );
  FastFedCablingHistograms( DaqMonitorBEInterface* );
  virtual ~FastFedCablingHistograms();
  
  void histoAnalysis( bool debug );
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

