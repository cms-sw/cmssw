#ifndef DQM_SiStripCommissioningSources_LatencyTask_h
#define DQM_SiStripCommissioningSources_LatencyTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <map>
#include <string>

/**
   @class LatencyTask
*/
class LatencyTask : public CommissioningTask {

 public:
  
  LatencyTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~LatencyTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  static std::map<std::string, HistoSet> timingMap_;
  HistoSet dummy_;
  HistoSet& timing_;

};

#endif // DQM_SiStripCommissioningSources_LatencyTask_h

