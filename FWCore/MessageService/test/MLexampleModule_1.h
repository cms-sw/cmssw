#ifndef FWCore_MessageService_test_MLexampleModule_1_h
#define FWCore_MessageService_test_MLexampleModule_1_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class MLexampleModule_1 : public edm::global::EDAnalyzer<> {
  public:
    explicit MLexampleModule_1(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const override;

  private:
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_MLexampleModule_1_h
