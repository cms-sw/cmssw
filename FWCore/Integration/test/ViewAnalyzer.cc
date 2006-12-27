#include "FWCore/Integration/test/ViewAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest 
{

  ViewAnalyzer::ViewAnalyzer(edm::ParameterSet const&) 
  { }

  ViewAnalyzer::~ViewAnalyzer() 
  { }

  void ViewAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) 
  {
  }
}

using edmtest::ViewAnalyzer;
DEFINE_FWK_MODULE(ViewAnalyzer);
