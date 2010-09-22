#ifndef FWCore_MessageService_test_UnitTestClient_Ad_h
#define FWCore_MessageService_test_UnitTestClient_Ad_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_Ad
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_Ad( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_Ad()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_Ad_h
