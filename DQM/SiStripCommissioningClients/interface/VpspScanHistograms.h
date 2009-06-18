#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMOldReceiver;
class DQMStore;

class VpspScanHistograms : public virtual CommissioningHistograms {

 public:
  
  VpspScanHistograms( const edm::ParameterSet& pset, DQMOldReceiver* );
  VpspScanHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~VpspScanHistograms();
  
  void histoAnalysis( bool debug );

  void printAnalyses(); // override

};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


