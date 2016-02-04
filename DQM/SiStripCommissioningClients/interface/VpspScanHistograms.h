#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class VpspScanHistograms : public virtual CommissioningHistograms {

 public:
  
  VpspScanHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~VpspScanHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


