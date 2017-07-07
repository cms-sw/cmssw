#ifndef FWCore_MessageService_test_UnitTestClient_F_h
#define FWCore_MessageService_test_UnitTestClient_F_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_F
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_F( edm::ParameterSet const & )
  { }

  
    ~UnitTestClient_F() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_F_h
