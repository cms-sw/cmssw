#include <cassert>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "ProdigalAnalyzer.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

namespace edmtest {
  ProdigalAnalyzer::ProdigalAnalyzer(edm::ParameterSet const& )
  {
  }

  void ProdigalAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    edm::Handle<Prodigal> h;
    assert(e.getByLabel("maker", h));
    assert(h.provenance()->parents().empty());    
  }

}
using edmtest::ProdigalAnalyzer;
DEFINE_FWK_MODULE(ProdigalAnalyzer);
