#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMOldReceiver;
class DQMStore;

class OptoScanHistograms : public virtual CommissioningHistograms {

 public:

  OptoScanHistograms( DQMOldReceiver* );
  OptoScanHistograms( DQMStore* );
  virtual ~OptoScanHistograms();
  
  void histoAnalysis( bool debug );
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistograms_H


