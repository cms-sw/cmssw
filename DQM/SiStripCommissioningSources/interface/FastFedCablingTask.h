#ifndef DQM_SiStripCommissioningSources_FastFedCablingTask_h
#define DQM_SiStripCommissioningSources_FastFedCablingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/** */
class FastFedCablingTask : public CommissioningTask {
public:
  FastFedCablingTask(DQMStore*, const FedChannelConnection&);
  ~FastFedCablingTask() override;

private:
  void book() override;
  void fill(const SiStripEventSummary&, const edm::DetSet<SiStripRawDigi>&) override;
  void update() override;

  HistoSet histo_;
};

#endif  // DQM_SiStripCommissioningSources_FastFedCablingTask_h
