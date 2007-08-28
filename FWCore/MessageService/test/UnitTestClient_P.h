#ifndef FWCore_MessageService_test_UnitTestClient_P_h
#define FWCore_MessageService_test_UnitTestClient_P_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UnitTestClient_P
  : public edm::EDAnalyzer
{
public:
  explicit
    UnitTestClient_P( edm::ParameterSet const & p) 
    : useLogFlush(true)
    , queueFillers(1)
  { 
    useLogFlush  = p.getUntrackedParameter<bool>("useLogFlush",  useLogFlush);
    queueFillers = p.getUntrackedParameter<int> ("queueFillers", queueFillers);
  }

  virtual
    ~UnitTestClient_P()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  bool useLogFlush;
  int  queueFillers;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_P_h
