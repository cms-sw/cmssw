#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class VpspScanHistograms : public CommissioningHistograms {

 public:
  
  /** */
  VpspScanHistograms( MonitorUserInterface* );
  /** */
  virtual ~VpspScanHistograms();

 private: // ----- private methods -----
  
  virtual void book( const std::vector<std::string>& me_list );
  virtual void update();
  
 private: // ----- private data members -----

  // One map entry per LLD channel...
  std::map< uint32_t, HistoSet > vpspApv0_;
  std::map< uint32_t, HistoSet > vpspApv1_;
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


