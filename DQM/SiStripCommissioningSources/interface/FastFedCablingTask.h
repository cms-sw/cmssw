#ifndef DQM_SiStripCommissioningSources_FastFedCablingTask_h
#define DQM_SiStripCommissioningSources_FastFedCablingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/** */
class FastFedCablingTask : public CommissioningTask {

 public:
  
  FastFedCablingTask( DQMStore*, const FedChannelConnection& );
  virtual ~FastFedCablingTask();
  
 private:
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  HistoSet histo_;
  
};

#endif // DQM_SiStripCommissioningSources_FastFedCablingTask_h

