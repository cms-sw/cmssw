#ifndef HLTriggerOffline_SUSYBSM_HltSusyExoPostProcessor_H
#define HLTriggerOffline_SUSYBSM_HltSusyExoPostProcessor_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HltSusyExoPostProcessor : public DQMEDHarvester {
public:
  HltSusyExoPostProcessor(const edm::ParameterSet &pset);
  ~HltSusyExoPostProcessor() override{};

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  MonitorElement *bookEffMEProfileFromTH1(TH1F *, std::string, DQMStore::IBooker &);
  std::string subDir_;
  bool mcFlag;
  std::vector<edm::ParameterSet> reco_parametersets;
  std::vector<edm::ParameterSet> mc_parametersets;
  std::vector<std::string> reco_dirs;
  std::vector<std::string> mc_dirs;
};

#endif
