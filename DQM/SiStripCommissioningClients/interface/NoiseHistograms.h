#ifndef DQM_SiStripCommissioningClients_NoiseHistograms_H
#define DQM_SiStripCommissioningClients_NoiseHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class NoiseHistograms : public virtual CommissioningHistograms {
public:
  NoiseHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~NoiseHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override
};

#endif  // DQM_SiStripCommissioningClients_NoiseHistograms_H
