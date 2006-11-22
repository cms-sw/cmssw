#ifndef DQM_SiStripCommissioningSources_OptoScanTask_H
#define DQM_SiStripCommissioningSources_OptoScanTask_H

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class OptoScanTask
*/
class OptoScanTask : public CommissioningTask {

 public:
  
  OptoScanTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~OptoScanTask();
  
 private:
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  void locateTicks( const edm::DetSet<SiStripRawDigi>& scope_mode_data,
		    std::pair< uint16_t, uint16_t >& digital_range, 
		    bool first_tick_only = false );
  
  /** "Histo sets" for the various gain settings and digital 0 and 1
      levels. First index is gain (0->3) and the second index is the
      digital level ("0" or "1"). */
  std::vector< std::vector<HistoSet> > opto_;
  
  uint16_t nBins_;
  
};

#endif // DQM_SiStripCommissioningSources_OptoScanTask_H

