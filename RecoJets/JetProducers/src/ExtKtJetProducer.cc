// File: FastJetProducer.cc
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Creation Date:  Feb. 1 2007 Initial version.
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/ExtKtJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"

//  Wrapper around ktjet-package (http://projects.hepforge.org/ktjet)
//  See Reference: Comp. Phys. Comm. vol 153/1 85-96 (2003)
//  Also:  http://www.arxiv.org/abs/hep-ph/0210022
//  this package is included in the external CMSSW-dependencies
//  License of package: GPL

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  ExtKtJetProducer::ExtKtJetProducer(const edm::ParameterSet& conf)
    : BaseJetProducer (conf), alg_(conf)

  {}

  // run algorithm itself
  bool ExtKtJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
				       JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);

    return true;
  }
}
