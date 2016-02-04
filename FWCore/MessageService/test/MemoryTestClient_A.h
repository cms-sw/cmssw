#ifndef FWCore_MessageService_test_MemoryTestClient_A_h
#define FWCore_MessageService_test_MemoryTestClient_A_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <vector>

// MemoryTestClient_A is used for testing JObReprt outputs from
// the MemoryService

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class MemoryTestClient_A
  : public edm::EDAnalyzer
{
public:
  explicit
    MemoryTestClient_A( edm::ParameterSet const & );

  virtual
    ~MemoryTestClient_A()
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
  char* last_allocation;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_MemoryTestClient_A_h
