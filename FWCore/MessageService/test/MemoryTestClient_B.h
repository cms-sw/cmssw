#ifndef FWCore_MessageService_test_MemoryTestClient_B_h
#define FWCore_MessageService_test_MemoryTestClient_B_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <vector>

// MemoryTestClient_B is used for testing JObReprt outputs from
// the MemoryService

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class MemoryTestClient_B : public edm::one::EDAnalyzer<> {
  public:
    explicit MemoryTestClient_B(edm::ParameterSet const&);

    void analyze(edm::Event const& e, edm::EventSetup const& c) final;

  private:
    static int nevent;
    std::vector<double> memoryPattern;
    void initializeMemoryPattern(int pattern);
    double vsize;
    edm::propagate_const<char*> last_Allocation;
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_MemoryTestClient_B_h
