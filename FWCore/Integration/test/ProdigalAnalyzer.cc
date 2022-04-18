#include <cassert>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "ProdigalAnalyzer.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

namespace edmtest {
  ProdigalAnalyzer::ProdigalAnalyzer(edm::ParameterSet const&) { consumes<Prodigal>(edm::InputTag{"maker"}); }

  void ProdigalAnalyzer::analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const&) const {
    edm::Handle<Prodigal> h;
    assert(e.getByLabel("maker", h));
    assert(h.provenance()->productProvenance()->parentage().parents().empty());
  }

}  // namespace edmtest
using edmtest::ProdigalAnalyzer;
DEFINE_FWK_MODULE(ProdigalAnalyzer);
