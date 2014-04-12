#ifndef DQM_SiStripCommissioningSources_PedsAndNoiseTask_h
#define DQM_SiStripCommissioningSources_PedestalsTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class PedestalsTask
*/
class PedestalsTask : public CommissioningTask {

 public:
  
  PedestalsTask( DQMStore*, const FedChannelConnection& );
  virtual ~PedestalsTask();
  
 private:
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  std::vector<HistoSet> peds_;
  std::vector<HistoSet> cm_;
  
};

#endif // DQM_SiStripCommissioningSources_PedestalsTask_h

