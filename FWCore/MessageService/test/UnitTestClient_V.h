#ifndef FWCore_MessageService_test_UnitTestClient_V_h
#define FWCore_MessageService_test_UnitTestClient_V_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UTC_V1
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_V1( edm::ParameterSet const & p) : ev(0)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 99);
  }

  virtual
    ~UTC_V1()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

  virtual void beginJob ();
  virtual void beginRun (edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock
  		(edm::LuminosityBlock const&, edm::EventSetup const&);



private:
  int identifier;
  int ev;
};

class UTC_V2
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_V2( edm::ParameterSet const & p) : ev(0)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 98);
  }

  virtual
    ~UTC_V2()
  { }

  virtual
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                );

private:
  int identifier;
  int ev;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_T_h
