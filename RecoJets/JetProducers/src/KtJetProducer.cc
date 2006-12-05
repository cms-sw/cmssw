// File: KtJetProducer.cc
// Description:  see KtJetProducer.h
// Author:  Fernando Varela Rodriguez, Boston University
// Creation Date:  Apr. 22 2005 Initial version.
// Revisions:  R. Harris, 19-Oct-2005, modified to use real CaloTowers from Jeremy Mans
// Revisions:  F.Ratnikov, 8-Mar-2006, accommodate Candidate model
// $Id: KtJetProducer.cc,v 1.17 2006/08/22 22:11:41 fedor Exp $
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  KtJetProducer::KtJetProducer(const edm::ParameterSet& conf)
  : BaseJetProducer (conf),
    alg_(conf.getParameter<int>("ktAngle"),
	 1, // use E-recombination
	 conf.getParameter<double>("ktECut"),
	 conf.getParameter<double>("ktRParam"))
  {
    // branch alias
    char label [32];
    sprintf (label, "KT%d%s", 
	     int (floor (conf.getParameter<double>("ktRParam") * 10. + 0.5)),
	     jetType ().c_str());
    initBranch (label);
  }

  // run algorithm itself
  bool KtJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}
