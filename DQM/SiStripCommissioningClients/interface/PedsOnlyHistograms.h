#ifndef DQM_SiStripCommissioningClients_PedsOnlyHistograms_H
#define DQM_SiStripCommissioningClients_PedsOnlyHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class PedsOnlyHistograms : public virtual CommissioningHistograms {

 public:
  
  PedsOnlyHistograms( const edm::ParameterSet& pset, DQMStore* );
  ~PedsOnlyHistograms() override;
  
  void histoAnalysis( bool debug ) override;

  void printAnalyses() override; // override

};

#endif // DQM_SiStripCommissioningClients_PedsOnlyHistograms_H
