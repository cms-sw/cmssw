#ifndef DQM_SiStripCommissioningClients_PedsOnlyHistograms_H
#define DQM_SiStripCommissioningClients_PedsOnlyHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQMServices/Core/interface/DQMStore.h"

class PedsOnlyHistograms : public virtual CommissioningHistograms {
public:
  PedsOnlyHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~PedsOnlyHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override
};

#endif  // DQM_SiStripCommissioningClients_PedsOnlyHistograms_H
