#ifndef FWCore_MessageService_test_UnitTestClient_T_h
#define FWCore_MessageService_test_UnitTestClient_T_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UTC_T1
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_T1( edm::ParameterSet const & p) : ev(0)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 99);
  }

  virtual
    ~UTC_T1()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  int identifier;
  int ev;
};

class UTC_T2
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_T2( edm::ParameterSet const & p) : ev(0)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 98);
  }

  virtual
    ~UTC_T2()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  int identifier;
  int ev;
  void printLES(std::vector<edm::ErrorSummaryEntry> const & v);
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_T_h
