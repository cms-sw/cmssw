#ifndef FWCore_MessageService_test_UnitTestClient_R_h
#define FWCore_MessageService_test_UnitTestClient_R_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_R
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_R( edm::ParameterSet const & )
  { }

  
    ~UnitTestClient_R() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_A_h
