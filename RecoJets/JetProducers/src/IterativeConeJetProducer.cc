#include <memory>

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Handle.h"

#include "RecoJets/JetProducers/interface/IterativeConeJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms {

  IterativeConeJetProducer::IterativeConeJetProducer(edm::ParameterSet const& conf):
    BaseJetProducer (conf),
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("coneRadius"),
	 conf.getParameter<double>("towerThreshold"))
  {
    // branch alias
    char label [32];
    sprintf (label, "IC%d%s", 
	     int (floor (conf.getParameter<double>("coneRadius") * 10. + 0.5)), 
	     jetType ().c_str());
    initBranch (label);
  }

  // run algorithm itself
  bool IterativeConeJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}


