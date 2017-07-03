#ifndef FWCore_MessageService_test_UnitTestClient_E_h
#define FWCore_MessageService_test_UnitTestClient_E_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_E
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_E( edm::ParameterSet const & )
  { }

  
    ~UnitTestClient_E() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_E_h
