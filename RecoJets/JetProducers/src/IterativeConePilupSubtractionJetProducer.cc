#include "IterativeConePilupSubtractionJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms {

  IterativeConePilupSubtractionJetProducer::IterativeConePilupSubtractionJetProducer(edm::ParameterSet const& conf):
    BasePilupSubtractionJetProducer (conf),
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("coneRadius"))
  {}

  // run algorithm itself
  bool IterativeConePilupSubtractionJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}


