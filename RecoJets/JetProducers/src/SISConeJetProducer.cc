// File: SISConeJetProducer.cc
// Description:  see SISConeJetProducer.h
// Author:  Fedor Ratnikov, Maryland, June 30, 2007
// $Id: SISConeJetProducer.cc,v 1.1 2007/06/30 17:24:07 fedor Exp $
//
//--------------------------------------------
#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "SISConeJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.

  SISConeJetProducer::SISConeJetProducer(edm::ParameterSet const& conf):
    BaseJetProducer (conf),
    alg_(conf)
  {}


  // run algorithm itself
  bool SISConeJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}

