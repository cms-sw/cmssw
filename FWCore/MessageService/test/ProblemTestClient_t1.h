#ifndef FWCore_MessageService_test_ProblemTestClient_t1_h
#define FWCore_MessageService_test_ProblemTestClient_t1_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class ProblemTestClient_t1
  : public edm::EDAnalyzer
{
public:
  explicit
    ProblemTestClient_t1( edm::ParameterSet const & )
  { }

  virtual
    ~ProblemTestClient_t1()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_ProblemTestClient_t1_h
