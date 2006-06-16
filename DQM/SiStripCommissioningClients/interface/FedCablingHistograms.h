#ifndef DQM_SiStripCommissioningClients_FedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class FedCablingHistograms : public CommissioningHistograms {

 public:
  
  /** */
  FedCablingHistograms( MonitorUserInterface* );
  /** */
  virtual ~FedCablingHistograms();

 private: // ----- private methods -----
  
  virtual void book( const std::vector<std::string>& me_list );
  virtual void update();
  
 private: // ----- private data members -----
  
  std::map< uint32_t, HistoSet > cabling_;
  
};

#endif // DQM_SiStripCommissioningClients_FedCablingHistograms_H


