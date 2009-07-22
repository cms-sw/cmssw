#ifndef DQM_SiStripCommissioningClients_NoiseHistograms_H
#define DQM_SiStripCommissioningClients_NoiseHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMOldReceiver;
class DQMStore;

class NoiseHistograms : public virtual CommissioningHistograms {

 public:
  
  NoiseHistograms( DQMOldReceiver* );
  NoiseHistograms( DQMStore* );
  virtual ~NoiseHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_NoiseHistograms_H
