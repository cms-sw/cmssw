#ifndef DQM_SiStripCommissioningClients_PedsFullNoiseHistograms_H
#define DQM_SiStripCommissioningClients_PedsFullNoiseHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQMServices/Core/interface/DQMStore.h"


class PedsFullNoiseHistograms : public virtual CommissioningHistograms {
public:
  PedsFullNoiseHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~PedsFullNoiseHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override
};

#endif  // DQM_SiStripCommissioningClients_PedsFullNoiseHistograms_H
