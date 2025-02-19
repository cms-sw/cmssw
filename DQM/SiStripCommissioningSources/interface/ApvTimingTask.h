#ifndef DQM_SiStripCommissioningSources_ApvTimingTask_h
#define DQM_SiStripCommissioningSources_ApvTimingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class ApvTimingTask
*/
class ApvTimingTask : public CommissioningTask {
  
 public:
  
  ApvTimingTask( DQMStore*, const FedChannelConnection& );
  virtual ~ApvTimingTask();
  
 private:
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  HistoSet timing_;

  uint16_t nSamples_;
  uint16_t nFineDelays_;
  uint16_t nBins_;
  
};

#endif // DQM_SiStripCommissioningSources_ApvTimingTask_h

