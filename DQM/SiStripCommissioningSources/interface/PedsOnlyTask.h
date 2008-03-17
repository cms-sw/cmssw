#ifndef DQM_SiStripCommissioningSources_PedsOnlyTask_h
#define DQM_SiStripCommissioningSources_PedsOnlyTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class PedsOnlyTask
*/
class PedsOnlyTask : public CommissioningTask {

 public:
  
  PedsOnlyTask( DQMStore*, const FedChannelConnection& );
  virtual ~PedsOnlyTask();
  
 private:
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  std::vector<HistoSet> peds_;
  
};

#endif // DQM_SiStripCommissioningSources_PedsOnlyTask_h

