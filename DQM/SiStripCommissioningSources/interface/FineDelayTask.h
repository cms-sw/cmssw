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
  ~FineDelayTask() override;
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) override;
  void update() override;
  
  static HistoSet timing_;
  static MonitorElement * mode_;

};

#endif // DQM_SiStripCommissioningSources_FineDelayTask_h

