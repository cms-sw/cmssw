// This test module will look for history information in event data.

#include <iostream>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest 
{
  class TestHistoryKeeping : public edm::EDAnalyzer
  {
  public:

    explicit TestHistoryKeeping(edm::ParameterSet const& pset);
    virtual ~TestHistoryKeeping();
    
    virtual void analyze(edm::Event const& e, edm::EventSetup const&);

    virtual void beginRun(edm::Run const& r, edm::EventSetup const&);
    virtual void endRun(edm::Run const& r, edm::EventSetup const&);

  private:
  }; // class TestHistoryKeeping

  //--------------------------------------------------------------------
  //
  // Implementation details

  TestHistoryKeeping::TestHistoryKeeping(edm::ParameterSet const& pset)
  {
  }

  TestHistoryKeeping::~TestHistoryKeeping() {}

  void
  TestHistoryKeeping::analyze(edm::Event const& ev, edm::EventSetup const&)
  {
    // Make sure the configuration for each process can be found.
    std::vector<std::string> expected_processes;
    expected_processes.push_back("HLT");
    expected_processes.push_back("PROD");
    expected_processes.push_back("TEST");
    
    for (std::vector<std::string>::const_iterator
	   i = expected_processes.begin(),
	   e = expected_processes.end();
	 i != e;
	 ++i)
      {
	edm::ParameterSet ps;
	assert(ev.getProcessParameterSet(*i, ps));
	assert(!ps.empty());
	assert(ps.getParameter<std::string>("@process_name") == *i);
      }
  }

  void
  TestHistoryKeeping::beginRun(edm::Run const&, edm::EventSetup const&)
  {
  }

  void
  TestHistoryKeeping::endRun(edm::Run const&, edm::EventSetup const& )
  {
    // Nothing to do.
  }

} // namespace edmtest

using edmtest::TestHistoryKeeping;
DEFINE_FWK_MODULE(TestHistoryKeeping);
