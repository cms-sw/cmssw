#ifndef FWCore_MessageService_test_MemoryTestClient_B_h
#define FWCore_MessageService_test_MemoryTestClient_B_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <vector>

// MemoryTestClient_B is used for testing JObReprt outputs from
// the MemoryService

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class MemoryTestClient_B
  : public edm::EDAnalyzer
{
public:
  explicit
    MemoryTestClient_B( edm::ParameterSet const & );

  virtual
    ~MemoryTestClient_B()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  static int nevent;
  std::vector<double> memoryPattern;
  void initializeMemoryPattern(int pattern);
  double vsize;
  char* last_Allocation;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_MemoryTestClient_B_h
