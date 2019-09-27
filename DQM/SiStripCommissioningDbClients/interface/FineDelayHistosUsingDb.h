
#ifndef DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <string>
#include <map>

class TrackerGeometry;

class FineDelayHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
public:
  FineDelayHistosUsingDb(const edm::ParameterSet& pset, DQMStore*, SiStripConfigDb* const);

  ~FineDelayHistosUsingDb() override;

  void configure(const edm::ParameterSet&, const edm::EventSetup&) override;

  void uploadConfigurations() override;

private:
  bool update(SiStripConfigDb::DeviceDescriptionsRange);

  void update(SiStripConfigDb::FedDescriptionsRange);

  void create(SiStripConfigDb::AnalysisDescriptionsV&, Analysis) override;

  void computeDelays();

  std::map<unsigned int, float> delays_;

  const TrackerGeometry* tracker_;

  bool cosmic_;
};

#endif  // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
