#ifndef FWCore_MessageService_test_UnitTestClient_N_h
#define FWCore_MessageService_test_UnitTestClient_N_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_N
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_N( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_N()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_N_h
