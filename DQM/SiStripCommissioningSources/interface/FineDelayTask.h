#ifndef DQM_SiStripCommissioningSources_FineDelayTask_h
#define DQM_SiStripCommissioningSources_FineDelayTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <map>
#include <string>

/**
   @class FineDelayTask
*/
class FineDelayTask : public CommissioningTask {

 public:
  
  FineDelayTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~FineDelayTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  static std::map<std::string, HistoSet> timingMap_;
  HistoSet dummy_;
  HistoSet& timing_;

  uint16_t nBins_;
  float fiberLengthCorrection_;

};

#endif // DQM_SiStripCommissioningSources_FineDelayTask_h

