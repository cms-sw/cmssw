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
  
  /** */
  virtual void histoAnalysis();
  /** */
  virtual void createSummaryHistos();
  /** */
  virtual void createTrackerMap() {;}
  /** */
  virtual void uploadToConfigDb();
  
 private: // ----- private methods -----
  
  virtual void book( const std::vector<std::string>& me_list );
  virtual void update();

 private: // ----- private data members -----
  
  // One map entry per LLD channel (key = control key)...
  std::map< uint32_t, HistoSet > timing_;
  
  // Key = control key, data = timing delay [ns]
  std::map< uint32_t, uint32_t > delays_;
  
  MonitorElement* profile_;
  MonitorElement* summary_;
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H


