#ifndef FWCore_MessageService_test_UnitTestClient_B_h
#define FWCore_MessageService_test_UnitTestClient_B_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

// UnitTestClient_B is used for testing LogStatistics and the reset behaviors
// of statistics destinations.

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_B
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_B( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_B()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  static int nevent;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_B_h
