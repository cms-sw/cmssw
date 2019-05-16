#ifndef DQM_SiStripCommissioningClients_SamplingHistograms_H
#define DQM_SiStripCommissioningClients_SamplingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SamplingSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/SamplingAnalysis.h"
#include "DQMServices/Core/interface/DQMStore.h"


class SamplingHistograms : virtual public CommissioningHistograms {
public:
  SamplingHistograms(const edm::ParameterSet& pset, DQMStore*, const sistrip::RunType& task = sistrip::APV_LATENCY);
  ~SamplingHistograms() override;

  void histoAnalysis(bool debug) override;

  void configure(const edm::ParameterSet&, const edm::EventSetup&) override;

private:
  float sOnCut_;

  int latencyCode_;
};

#endif  // DQM_SiStripCommissioningClients_SamplingHistograms_H
