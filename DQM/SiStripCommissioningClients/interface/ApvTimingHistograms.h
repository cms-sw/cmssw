#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class ApvTimingHistograms : public virtual CommissioningHistograms {
public:
  ApvTimingHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~ApvTimingHistograms() override;

  void histoAnalysis(bool debug) override;
};

#endif  // DQM_SiStripCommissioningClients_ApvTimingHistograms_H
