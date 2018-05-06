#ifndef DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H
#define DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class DaqScopeModeHistograms : public virtual CommissioningHistograms {

 public:
  
  DaqScopeModeHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~DaqScopeModeHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H
