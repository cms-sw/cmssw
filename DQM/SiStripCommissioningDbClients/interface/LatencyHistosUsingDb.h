
#ifndef DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <string>
#include <map>

class LatencyHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
public:
  LatencyHistosUsingDb(const edm::ParameterSet& pset, DQMStore*, SiStripConfigDb* const);

  ~LatencyHistosUsingDb() override;

  void uploadConfigurations() override;

  void configure(const edm::ParameterSet&, const edm::EventSetup&) override;

private:
  bool update(SiStripConfigDb::DeviceDescriptionsRange, SiStripConfigDb::FedDescriptionsRange);

  void create(SiStripConfigDb::AnalysisDescriptionsV&, Analysis) override;

  bool perPartition_;
};

#endif  // DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H
