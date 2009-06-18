#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMOldReceiver;
class DQMStore;

class ApvTimingHistograms : public virtual CommissioningHistograms {

 public:
  
  ApvTimingHistograms( const edm::ParameterSet& pset, DQMOldReceiver* );
  ApvTimingHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~ApvTimingHistograms();
  
  void histoAnalysis( bool debug );
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

