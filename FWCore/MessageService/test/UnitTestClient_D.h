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

  virtual
    ~UnitTestClient_D()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_D_h
