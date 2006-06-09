#ifndef DQM_SiStripCommissioningClients_FedTimingHistograms_H
#define DQM_SiStripCommissioningClients_FedTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class FedTimingHistograms : public CommissioningHistograms {

 public:
  
  /** */
  FedTimingHistograms( MonitorUserInterface* );
  /** */
  virtual ~FedTimingHistograms();

 private: // ----- private methods -----
  
  virtual void book( const std::vector<std::string>& me_list );
  virtual void update();
  
 private: // ----- private data members -----
  
  // One map entry per LLD channel...
  std::map< uint32_t, HistoSet > timing_;
  
};

#endif // DQM_SiStripCommissioningClients_FedTimingHistograms_H


