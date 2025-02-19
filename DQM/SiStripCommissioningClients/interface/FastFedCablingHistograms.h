#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class FastFedCablingHistograms : public virtual CommissioningHistograms {

 public:
  
  FastFedCablingHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~FastFedCablingHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

  void printSummary(); // override

};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

