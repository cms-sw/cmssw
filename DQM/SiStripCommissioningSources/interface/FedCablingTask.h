#ifndef DQM_SiStripCommissioningSources_FedCablingTask_h
#define DQM_SiStripCommissioningSources_FedCablingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class FedCablingTask
*/
class FedCablingTask : public CommissioningTask {

 public:
  
  FedCablingTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~FedCablingTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();

  /** HistoSet for FED cabling. First element contains histo info for
      FED id, second element contains histo info for FED channel. */
  vector<HistoSet> cabling_;
  
};

#endif // DQM_SiStripCommissioningSources_FedCablingTask_h
