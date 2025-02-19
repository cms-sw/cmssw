#ifndef FWCore_MessageService_test_UnitTestClient_J_h
#define FWCore_MessageService_test_UnitTestClient_J_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_J
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_J( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_J()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_J_h
