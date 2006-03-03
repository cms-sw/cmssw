#ifndef DQM_SiStripCommissioningSources_ApvTimingTask_h
#define DQM_SiStripCommissioningSources_ApvTimingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class ApvTimingTask
*/
class ApvTimingTask : public CommissioningTask {

 public:
  
  ApvTimingTask( DaqMonitorBEInterface*, const SiStripModule& );
  virtual ~ApvTimingTask();
  
 private:

  virtual void book( const SiStripModule& );
  virtual void fill( const vector<StripDigi>& );
  virtual void update();
  
  vector<HistoSet> timing_;

};

#endif // DQM_SiStripCommissioningSources_ApvTimingTask_h

