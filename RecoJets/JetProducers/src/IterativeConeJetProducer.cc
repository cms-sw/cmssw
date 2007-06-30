#include "FWCore/Framework/interface/MakerMacros.h"
#include "IterativeConeJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms {

  IterativeConeJetProducer::IterativeConeJetProducer(edm::ParameterSet const& conf):
    BaseJetProducer (conf),
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("coneRadius"))
  {}

  // run algorithm itself
  bool IterativeConeJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}



