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

  virtual
    ~UnitTestClient_R()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_A_h
