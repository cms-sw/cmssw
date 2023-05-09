#ifndef Integration_ProdigalAnalyzer_h
#define Integration_ProdigalAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

namespace edmtest {

  class ProdigalAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ProdigalAnalyzer(edm::ParameterSet const& pset);
    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const final;
  };

}  // namespace edmtest

#endif
