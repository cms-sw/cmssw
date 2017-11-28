#ifndef FWCore_MessageService_test_UnitTestClient_H_h
#define FWCore_MessageService_test_UnitTestClient_H_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class UnitTestClient_H : public edm::EDAnalyzer {
  public:
    explicit UnitTestClient_H(edm::ParameterSet const&) {}

    virtual ~UnitTestClient_H() {}

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  private:
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_UnitTestClient_H_h
