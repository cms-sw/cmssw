#ifndef DQM_SiStripCommissioningSources_VpspScanTask_H
#define DQM_SiStripCommissioningSources_VpspScanTask_H

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class VpspScanTask
*/
class VpspScanTask : public CommissioningTask {

 public:
  
  VpspScanTask( DQMStore*, const FedChannelConnection& );
  ~VpspScanTask() override;
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) override;
  void update() override;
  
  std::vector<HistoSet> vpsp_;
  
};

#endif // DQM_SiStripCommissioningSources_VpspScanTask_H
