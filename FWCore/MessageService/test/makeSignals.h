#ifndef FWCore_MessageService_test_makeSignals_h
#define FWCore_MessageService_test_makeSignals_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class makeSignals
  : public edm::EDAnalyzer
{
public:
  explicit
    makeSignals( edm::ParameterSet const & ) { }

  virtual
    ~makeSignals() { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};

}  // namespace edmtest


#endif  // FWCore_MessageService_test_makeSignals_h
