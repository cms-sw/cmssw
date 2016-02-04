#ifndef FWCore_MessageService_test_UnitTestClient_O_h
#define FWCore_MessageService_test_UnitTestClient_O_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_O
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_O( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_O()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_O_h
