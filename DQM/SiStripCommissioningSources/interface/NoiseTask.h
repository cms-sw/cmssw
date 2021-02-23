#ifndef DQM_SISTRIPCOMMISSIONINGSOURCES_NOISETASK_H
#define DQM_SISTRIPCOMMISSIONINGSOURCES_NOISETASK_H

#include <vector>

#include "DataFormats/Common/interface/DetSet.h"
#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

// Forward Declarations
class ApvAnalysisFactory;
class FedChannelConnection;
class SiStripEventSummary;
class SiStripRawDigi;

/**
 *  @class NoiseTask
 */
class NoiseTask : public CommissioningTask {
public:
  NoiseTask(DQMStore *,
            const FedChannelConnection &,
            edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalToken,
            edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken);
  ~NoiseTask() override;

private:
  void book() override;
  void fill(const SiStripEventSummary &, const edm::DetSet<SiStripRawDigi> &) override;
  void update() override;

  std::vector<HistoSet> peds_;
  std::vector<HistoSet> cm_;

  ApvAnalysisFactory *pApvFactory_;
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
};

#endif  // DQM_SISTRIPCOMMISSIONINGSOURCES_NOISETASK_H
