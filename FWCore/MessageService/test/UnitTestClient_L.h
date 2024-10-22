#ifndef FWCore_MessageService_test_UnitTestClient_L_h
#define FWCore_MessageService_test_UnitTestClient_L_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class UnitTestClient_L : public edm::one::EDAnalyzer<> {
  public:
    explicit UnitTestClient_L(edm::ParameterSet const&) {}

    void analyze(edm::Event const& e, edm::EventSetup const& c) final;

  private:
  };

  class UnitTestClient_L1 : public edm::one::EDAnalyzer<> {
  public:
    explicit UnitTestClient_L1(edm::ParameterSet const&) {}

    void analyze(edm::Event const& e, edm::EventSetup const& c) final;

  private:
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_UnitTestClient_L_h
