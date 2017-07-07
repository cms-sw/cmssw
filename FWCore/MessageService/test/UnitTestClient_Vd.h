#ifndef FWCore_MessageService_test_UnitTestClient_Vd_h
#define FWCore_MessageService_test_UnitTestClient_Vd_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class UTC_Vd1
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_Vd1( edm::ParameterSet const & p) : ev(0)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 99);
  }

  
    ~UTC_Vd1() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

  void beginJob () override;
  void beginRun (edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock
  		(edm::LuminosityBlock const&, edm::EventSetup const&) override;



private:
  int identifier;
  int ev;
};

class UTC_Vd2
  : public edm::EDAnalyzer
{
public:
  explicit
    UTC_Vd2( edm::ParameterSet const & p) : ev(0)
  { 
    identifier = p.getUntrackedParameter<int> ("identifier", 98);
  }

  
    ~UTC_Vd2() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
  int identifier;
  int ev;
};


}  // namespace edmtest


#endif  // FWCore_MessageService_test_UnitTestClient_Vd_h
