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

  
    ~MLexampleModule_1() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_MLexampleModule_1_h
