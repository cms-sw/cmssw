#ifndef FWCore_MessageService_test_UnitTestClient_S_h
#define FWCore_MessageService_test_UnitTestClient_S_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UTC_S1
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_S1( edm::ParameterSet const & p)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 99);
    edm::GroupLogStatistics("grouped_cat");  
  }

  virtual
    ~UTC_S1()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  int identifier;
  static bool enableNotYetCalled;
  static int n;
};

class UTC_S2
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_S2( edm::ParameterSet const & p)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 98);
  }

  virtual
    ~UTC_S2()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  int identifier;
  static int n;
};

class UTC_SUMMARY
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_SUMMARY( edm::ParameterSet const &)
  { 
  }

  virtual
    ~UTC_SUMMARY()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_S_h
