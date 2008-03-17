#ifndef DQM_SiStripCommissioningSources_NoiseTask_h
#define DQM_SiStripCommissioningSources_NoiseTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class NoiseTask
*/
class NoiseTask : public CommissioningTask {

 public:
  
  NoiseTask( DQMStore*, const FedChannelConnection& );
  virtual ~NoiseTask();
  
 private:
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  std::vector<HistoSet> peds_;
  std::vector<HistoSet> cm_;
  
};

#endif // DQM_SiStripCommissioningSources_NoiseTask_h

