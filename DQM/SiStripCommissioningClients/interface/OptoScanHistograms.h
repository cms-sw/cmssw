#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class OptoScanHistograms : public virtual CommissioningHistograms {

 public:

  OptoScanHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~OptoScanHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_OptoScanHistograms_H


