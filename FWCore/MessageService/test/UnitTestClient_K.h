#ifndef FWCore_MessageService_test_UnitTestClient_K_h
#define FWCore_MessageService_test_UnitTestClient_K_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_K
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_K( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_K()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_K_h
