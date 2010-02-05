#ifndef HLTriggerOffline_SUSYBSM_HltSusyExoPostProcessor_H
#define HLTriggerOffline_SUSYBSM_HltSusyExoPostProcessor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HltSusyExoPostProcessor : public edm::EDAnalyzer {
 public:
  HltSusyExoPostProcessor(const edm::ParameterSet& pset);
  ~HltSusyExoPostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endRun(edm::Run const&, edm::EventSetup const&);

  MonitorElement* bookEffMEProfileFromTH1(TH1F*,std::string);

  DQMStore* dqm;

 private:
  std::string subDir_;
  bool mcFlag;
  std::vector<edm::ParameterSet> reco_parametersets;
  std::vector<edm::ParameterSet> mc_parametersets;
  std::vector<std::string> reco_dirs;
  std::vector<std::string> mc_dirs;
  
};

#endif
