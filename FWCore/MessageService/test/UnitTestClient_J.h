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

  
    ~UnitTestClient_J() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_J_h
