#ifndef DQM_SiStripCommissioningSources_DaqScopeModeTask_h
#define DQM_SiStripCommissioningSources_DaqScopeModeTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class DaqScopeModeTask
*/
class DaqScopeModeTask : public CommissioningTask {

 public:
  
  DaqScopeModeTask( DQMStore*, const FedChannelConnection& );
  virtual ~DaqScopeModeTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  HistoSet scope_;

  uint16_t nBins_;

};

#endif // DQM_SiStripCommissioningSources_DaqScopeModeTask_h

