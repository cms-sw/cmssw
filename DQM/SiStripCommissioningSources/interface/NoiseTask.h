#ifndef DQM_SISTRIPCOMMISSIONINGSOURCES_NOISETASK_H
#define DQM_SISTRIPCOMMISSIONINGSOURCES_NOISETASK_H

#include <vector>

#include "DataFormats/Common/interface/DetSet.h"
#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

// Forward Declarations
class ApvAnalysisFactory;
class DQMStore;
class FedChannelConnection;
class SiStripEventSummary;
class SiStripRawDigi;

/**
 *  @class NoiseTask
 */
class NoiseTask : public CommissioningTask {
public:
  NoiseTask(DQMStore *, const FedChannelConnection &);
  ~NoiseTask() override;

private:
  void book() override;
  void fill(const SiStripEventSummary &, const edm::DetSet<SiStripRawDigi> &) override;
  void update() override;

  std::vector<HistoSet> peds_;
  std::vector<HistoSet> cm_;

  ApvAnalysisFactory *pApvFactory_;
};

#endif  // DQM_SISTRIPCOMMISSIONINGSOURCES_NOISETASK_H
