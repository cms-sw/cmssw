#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class FastFedCablingHistograms : public virtual CommissioningHistograms {
public:
  FastFedCablingHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~FastFedCablingHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override

  void printSummary() override;  // override
};

#endif  // DQM_SiStripCommissioningClients_FastFedCablingHistograms_H
