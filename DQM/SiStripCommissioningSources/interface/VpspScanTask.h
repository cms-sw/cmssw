#ifndef DQM_SiStripCommissioningSources_VpspScanTask_H
#define DQM_SiStripCommissioningSources_VpspScanTask_H

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class VpspScanTask
*/
class VpspScanTask : public CommissioningTask {

 public:
  
  VpspScanTask( DQMStore*, const FedChannelConnection& );
  virtual ~VpspScanTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  std::vector<HistoSet> vpsp_;
  
};

#endif // DQM_SiStripCommissioningSources_VpspScanTask_H
