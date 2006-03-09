#ifndef DQM_SiStripCommissioningSources_PedestalsTask_h
#define DQM_SiStripCommissioningSources_PedestalsTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class PedestalsTask
*/
class PedestalsTask : public CommissioningTask {

 public:
  
  PedestalsTask( DaqMonitorBEInterface*, const SiStripModule& );
  virtual ~PedestalsTask();
  
 private:

  virtual void book( const SiStripModule& );
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  HistoSet peds_;

};

#endif // DQM_SiStripCommissioningSources_PedestalsTask_h

