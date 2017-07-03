#ifndef FWCore_MessageService_test_UnitTestClient_C_h
#define FWCore_MessageService_test_UnitTestClient_C_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_C
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_C( edm::ParameterSet const & )
  { }

  
    ~UnitTestClient_C() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_A_h
