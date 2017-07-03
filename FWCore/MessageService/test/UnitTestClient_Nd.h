#define EDM_ML_DEBUG

#ifndef FWCore_MessageService_test_UnitTestClient_Nd_h
#define FWCore_MessageService_test_UnitTestClient_Nd_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_Nd
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_Nd( edm::ParameterSet const & )
  { }

  
    ~UnitTestClient_Nd() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_Nd_h
