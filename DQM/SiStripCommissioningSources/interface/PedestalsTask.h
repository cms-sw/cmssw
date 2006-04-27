#ifndef DQM_SiStripCommissioningSources_PedestalsTask_h
#define DQM_SiStripCommissioningSources_PedestalsTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class PedestalsTask
*/
class PedestalsTask : public CommissioningTask {

 public:
  
  PedestalsTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~PedestalsTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  vector<HistoSet> peds_;

  vector<uint32_t> vCommonMode0_;
  vector<uint32_t> vCommonMode1_;
  MonitorElement* meCommonMode0_;
  MonitorElement* meCommonMode1_;
  
};

#endif // DQM_SiStripCommissioningSources_PedestalsTask_h

