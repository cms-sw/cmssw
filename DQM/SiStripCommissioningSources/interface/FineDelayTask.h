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
  
  FineDelayTask( DQMStore*, const FedChannelConnection& );
  virtual ~FineDelayTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  static HistoSet timing_;
  static MonitorElement * mode_;

};

#endif // DQM_SiStripCommissioningSources_FineDelayTask_h

