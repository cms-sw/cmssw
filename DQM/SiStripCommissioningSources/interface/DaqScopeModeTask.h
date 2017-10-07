#ifndef DQM_SiStripCommissioningSources_DaqScopeModeTask_h
#define DQM_SiStripCommissioningSources_DaqScopeModeTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class DaqScopeModeTask
*/
class DaqScopeModeTask : public CommissioningTask {

 public:
  
  DaqScopeModeTask( DQMStore*, const FedChannelConnection& );
  ~DaqScopeModeTask() override;
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) override;
  void update() override;
  
  HistoSet scope_;

  uint16_t nBins_;

};

#endif // DQM_SiStripCommissioningSources_DaqScopeModeTask_h

