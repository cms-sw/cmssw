#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQMServices/Core/interface/DQMStore.h"


class OptoScanHistograms : public virtual CommissioningHistograms {
public:
  OptoScanHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~OptoScanHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override
};

#endif  // DQM_SiStripCommissioningClients_OptoScanHistograms_H
