#ifndef FWCore_MessageService_test_UnitTestClient_Q_h
#define FWCore_MessageService_test_UnitTestClient_Q_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UTC_Q1
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_Q1( edm::ParameterSet const & p)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 99);
    edm::GroupLogStatistics("timer");  // these lines would normally be in
    edm::GroupLogStatistics("trace");  // whatever service knows that
    				       // timer and trace should be groupd
				       // by moduels for statistics
  }

  
    ~UTC_Q1() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
  int identifier;
};

class UTC_Q2
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_Q2( edm::ParameterSet const & p)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 98);
  }

  
    ~UTC_Q2() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
  int identifier;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_Q_h
