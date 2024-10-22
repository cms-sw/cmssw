#ifndef FWCore_MessageService_test_PSetTestClient_A_h
#define FWCore_MessageService_test_PSetTestClient_A_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class PSetTestClient_A : public edm::global::EDAnalyzer<> {
  public:
    explicit PSetTestClient_A(edm::ParameterSet const& p);

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const final;

  private:
    edm::ParameterSet a;
    edm::ParameterSet b;
    int xa;
    int xb;
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_PSetTestClient_A_h
