#ifndef DQM_SiStripCommissioningSources_PedestalsTask_h
#define DQM_SiStripCommissioningSources_PedestalsTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class PedestalsTask
*/
class PedestalsTask : public CommissioningTask {
public:
  PedestalsTask(DQMStore*, const FedChannelConnection&);
  ~PedestalsTask() override;

private:
  void book() override;
  void fill(const SiStripEventSummary&, const edm::DetSet<SiStripRawDigi>&) override;
  void update() override;

  std::vector<HistoSet> peds_;
  std::vector<HistoSet> cm_;
};

#endif  // DQM_SiStripCommissioningSources_PedestalsTask_h
