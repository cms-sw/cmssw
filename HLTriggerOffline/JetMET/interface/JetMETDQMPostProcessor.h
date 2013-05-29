#ifndef HLTriggerOffline_JetMET_JetMETDQMPosProcessor_H
#define HLTriggerOffline_JetMET_JetMETDQMPosProcessor_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


class JetMETDQMPostProcessor : public edm::EDAnalyzer {
 public:
  JetMETDQMPostProcessor(const edm::ParameterSet& pset);
  ~JetMETDQMPostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endRun(edm::Run const&, edm::EventSetup const&);
  void endJob();

 private:
  std::string subDir_;

};

#endif
