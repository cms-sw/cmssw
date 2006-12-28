#ifndef FWCore_MessageService_test_UnitTestClient_G_h
#define FWCore_MessageService_test_UnitTestClient_G_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_G
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_G( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_G()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_A_h
