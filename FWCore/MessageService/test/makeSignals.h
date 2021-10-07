#ifndef FWCore_MessageService_test_makeSignals_h
#define FWCore_MessageService_test_makeSignals_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class makeSignals : public edm::global::EDAnalyzer<> {
  public:
    explicit makeSignals(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const final;

  private:
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_makeSignals_h
