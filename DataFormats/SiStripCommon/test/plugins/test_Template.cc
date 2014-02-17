// Last commit: $Id: test_Template.cc,v 1.3 2010/01/07 11:21:03 lowette Exp $

#include "DataFormats/SiStripCommon/test/plugins/test_Template.h"
#include "FWCore/Framework/interface/Event.h" 
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
test_Template::test_Template( const edm::ParameterSet& pset ) 
{
  LogTrace(mlTest_)
    << "[test_Template::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
test_Template::~test_Template() {
  LogTrace(mlTest_)
    << "[test_Template::" << __func__ << "]"
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
void test_Template::analyze( const edm::Event& event, 
			     const edm::EventSetup& setup ) {
  LogTrace(mlTest_) 
    << "[test_Template::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}


