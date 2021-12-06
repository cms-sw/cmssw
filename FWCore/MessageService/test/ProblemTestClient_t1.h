#ifndef FWCore_MessageService_test_ProblemTestClient_t1_h
#define FWCore_MessageService_test_ProblemTestClient_t1_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class ProblemTestClient_t1 : public edm::global::EDAnalyzer<> {
  public:
    explicit ProblemTestClient_t1(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const final;

  private:
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_ProblemTestClient_t1_h
