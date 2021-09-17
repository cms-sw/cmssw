#ifndef TRACKINGDQMCLIENTHEAVYIONS_H
#define TRACKINGDQMCLIENTHEAVYIONS_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <set>
#include <string>
#include <vector>
#include <TH1.h>
#include <TEfficiency.h>

class TrackingDQMClientHeavyIons : public DQMEDHarvester {
public:
  TrackingDQMClientHeavyIons(const edm::ParameterSet& pset);
  ~TrackingDQMClientHeavyIons() override{};

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  unsigned int verbose_;
  bool isWildcardUsed_;
  bool resLimitedFit_;

  std::string histName;
  std::string TopFolder_;
  MonitorElement* DCAStats;
  std::string outputFileName_;
};
#endif
