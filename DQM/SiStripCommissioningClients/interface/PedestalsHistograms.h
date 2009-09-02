#ifndef DQM_SiStripCommissioningClients_PedestalsHistograms_H
#define DQM_SiStripCommissioningClients_PedestalsHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMOldReceiver;
class DQMStore;

class PedestalsHistograms : public virtual CommissioningHistograms {

 public:
  
  PedestalsHistograms( DQMOldReceiver* );
  PedestalsHistograms( DQMStore* );
  virtual ~PedestalsHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
