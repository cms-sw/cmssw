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
  
  LatencyTask( DQMStore*, const FedChannelConnection& );
  virtual ~LatencyTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  static HistoSet timing_;
  static HistoSet cluster_;
  HistoSet timingPartition_;
  HistoSet clusterPartition_;
  int firstReading_;

};

#endif // DQM_SiStripCommissioningSources_LatencyTask_h

