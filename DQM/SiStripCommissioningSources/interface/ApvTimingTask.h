#ifndef DQM_SiStripCommissioningSources_ApvTimingTask_h
#define DQM_SiStripCommissioningSources_ApvTimingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   @class ApvTimingTask
*/
class ApvTimingTask : public CommissioningTask {
public:
  ApvTimingTask(DQMStore*, const FedChannelConnection&);
  ~ApvTimingTask() override;

private:
  void book() override;
  void fill(const SiStripEventSummary&, const edm::DetSet<SiStripRawDigi>&) override;
  void update() override;

  HistoSet timing_;

  uint16_t nSamples_;
  uint16_t nFineDelays_;
  uint16_t nBins_;
};

#endif  // DQM_SiStripCommissioningSources_ApvTimingTask_h
