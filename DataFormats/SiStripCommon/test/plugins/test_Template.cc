// system includes
#include <iostream>
#include <sstream>

// user includes
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**
   @class test_Template 
   @author R.Bainbridge
   @brief Simple class that tests Template.
*/
class test_Template : public edm::one::EDAnalyzer<> {
public:
  test_Template(const edm::ParameterSet&);
  ~test_Template();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

private:
};

using namespace sistrip;

// -----------------------------------------------------------------------------
//
test_Template::test_Template(const edm::ParameterSet& pset) {
  LogTrace(mlTest_) << "[test_Template::" << __func__ << "]"
                    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
test_Template::~test_Template() {
  LogTrace(mlTest_) << "[test_Template::" << __func__ << "]"
                    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void test_Template::beginJob() {
  std::stringstream ss;
  ss << "[test_Template::" << __func__ << "]"
     << " Initializing...";
  LogTrace(mlTest_) << ss.str();
}

// -----------------------------------------------------------------------------
//
void test_Template::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlTest_) << "[test_Template::" << __func__ << "]"
                    << " Analyzing run/event " << event.id().run() << "/" << event.id().event();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(test_Template);
