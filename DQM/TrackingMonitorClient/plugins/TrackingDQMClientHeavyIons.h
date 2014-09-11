#ifndef TRACKINGDQMCLIENTHEAVYIONS_H
#define TRACKINGDQMCLIENTHEAVYIONS_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include <set>
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <vector>
#include <TH1.h>
#include <RVersion.h>
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
#include <TEfficiency.h>
#else
#include <TGraphAsymmErrors.h>
#endif

class MonitorElement;

class TrackingDQMClientHeavyIons : public DQMEDHarvester
{
 public:
  TrackingDQMClientHeavyIons(const edm::ParameterSet& pset);
  ~TrackingDQMClientHeavyIons() {};

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

 private:
  unsigned int verbose_;
  bool isWildcardUsed_;
  bool resLimitedFit_;

  DQMStore* theDQM;
  std::string histName;
  std::string TopFolder_;
  MonitorElement* DCAStats;
  std::string outputFileName_;

};
#endif

