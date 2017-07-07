#ifndef Integration_ProdigalAnalyzer_h
#define Integration_ProdigalAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edmtest {

  class ProdigalAnalyzer : public edm::EDAnalyzer {
  public:
    explicit ProdigalAnalyzer(edm::ParameterSet const& pset);
    ~ProdigalAnalyzer() override {}
    void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  };

}

#endif
