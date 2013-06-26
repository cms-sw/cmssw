#ifndef FWCore_MessageService_test_UnitTestClient_L_h
#define FWCore_MessageService_test_UnitTestClient_L_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_L : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_L( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_L()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};

class UnitTestClient_L1 : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_L1( edm::ParameterSet const & )
  { }

  virtual
    ~UnitTestClient_L1()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest




#endif  // FWCore_MessageService_test_UnitTestClient_L_h
