#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include <vector>

class MonitorUserInterface;

class OptoScanHistograms : public CommissioningHistograms {

 public:

  /** */
  OptoScanHistograms( MonitorUserInterface* );
  /** */
  virtual ~OptoScanHistograms();

 private: // ----- private methods -----
  
  virtual void book( const std::vector<std::string>& me_list );
  virtual void update();
  
 private: // ----- private data members -----
  
  // One map entry per LLD channel...
  std::map< uint32_t, HistoSet > gain0digital0_;
  std::map< uint32_t, HistoSet > gain0digital1_;
  std::map< uint32_t, HistoSet > gain1digital0_;
  std::map< uint32_t, HistoSet > gain1digital1_;
  std::map< uint32_t, HistoSet > gain2digital0_;
  std::map< uint32_t, HistoSet > gain2digital1_;
  std::map< uint32_t, HistoSet > gain3digital0_;
  std::map< uint32_t, HistoSet > gain3digital1_;
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistograms_H


