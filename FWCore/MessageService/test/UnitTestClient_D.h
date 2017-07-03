#ifndef FWCore_MessageService_test_UnitTestClient_D_h
#define FWCore_MessageService_test_UnitTestClient_D_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_D
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_D( edm::ParameterSet const & )
  { }

  
    ~UnitTestClient_D() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_D_h
