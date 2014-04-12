#ifndef FWCore_MessageService_test_MLexampleModule_1_h
#define FWCore_MessageService_test_MLexampleModule_1_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class MLexampleModule_1
  : public edm::EDAnalyzer
{
public:
  explicit
    MLexampleModule_1( edm::ParameterSet const & )
  { }

  virtual
    ~MLexampleModule_1()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_MLexampleModule_1_h
