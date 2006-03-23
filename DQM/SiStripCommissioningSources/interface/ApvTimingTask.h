#ifndef DQM_SiStripCommissioningSources_ApvTimingTask_h
#define DQM_SiStripCommissioningSources_ApvTimingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class ApvTimingTask
*/
class ApvTimingTask : public CommissioningTask {

 public:
  
  ApvTimingTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~ApvTimingTask();
  
 private:

  virtual void book( const FedChannelConnection& );
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  HistoSet timing_;

};

#endif // DQM_SiStripCommissioningSources_ApvTimingTask_h

