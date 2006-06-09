#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;

class ApvTimingHistograms : public CommissioningHistograms {

 public:
  
  /** */
  ApvTimingHistograms( MonitorUserInterface* );
  /** */
  virtual ~ApvTimingHistograms();
  
  void createSummaryHistos(); 
  
 private: // ----- private methods -----
  
  virtual void book( const std::vector<std::string>& me_list );
  virtual void update();

 private: // ----- private data members -----
  
  // One map entry per LLD channel...
  std::map< uint32_t, HistoSet > timing_;
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H


