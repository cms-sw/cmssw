#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQMServices/Core/interface/DQMStore.h"

class VpspScanHistograms : public virtual CommissioningHistograms {
public:
  VpspScanHistograms(const edm::ParameterSet& pset, DQMStore*);
  ~VpspScanHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override
};

#endif  // DQM_SiStripCommissioningClients_VpspScanHistograms_H
