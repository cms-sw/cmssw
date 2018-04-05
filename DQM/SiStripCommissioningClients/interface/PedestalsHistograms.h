#ifndef DQM_SiStripCommissioningClients_PedestalsHistograms_H
#define DQM_SiStripCommissioningClients_PedestalsHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class PedestalsHistograms : public virtual CommissioningHistograms {

 public:
  
  PedestalsHistograms( const edm::ParameterSet& pset, DQMStore* );
  ~PedestalsHistograms() override;
  
  void histoAnalysis( bool debug ) override;

  void printAnalyses() override; // override

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
