#ifndef HLTriggerOffline_SUSYBSM_HltSusyExoPostProcessor_H
#define HLTriggerOffline_SUSYBSM_HltSusyExoPostProcessor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


class HltSusyExoPostProcessor : public edm::EDAnalyzer {
 public:
  HltSusyExoPostProcessor(const edm::ParameterSet& pset);
  ~HltSusyExoPostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endRun(edm::Run const&, edm::EventSetup const&);

 private:
  std::string subDir_;
  bool mcFlag;

};

#endif
