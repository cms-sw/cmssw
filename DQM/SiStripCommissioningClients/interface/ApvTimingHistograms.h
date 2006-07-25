#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
#include <map>

class MonitorUserInterface;
class SiStripSummary;

class ApvTimingHistograms : public CommissioningHistograms {

 public:

  // Key is SiStripControlKey
  typedef std::map< uint32_t, ApvTimingAnalysis::Monitorables > Data;
  
  /** */
  ApvTimingHistograms( MonitorUserInterface* );
  /** */
  virtual ~ApvTimingHistograms();
  
  /** */
  void histoAnalysis();

 private:
  
  Data data_;
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H


