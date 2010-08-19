#ifndef HLTriggerOffline_Egamma_EmDQMPosProcessor_H
#define HLTriggerOffline_Egamma_EmDQMPosProcessor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


class EmDQMPostProcessor : public edm::EDAnalyzer {
 public:
  EmDQMPostProcessor(const edm::ParameterSet& pset);
  ~EmDQMPostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endRun(edm::Run const&, edm::EventSetup const&);

 private:
  std::string subDir_;

};

#endif
