// File: CDFMidpointJetProducer.cc
// Description:  see CDFMidpointJetProducer.h
// Author:  M. Fedor Ratnikov, Maryland
// $Id: CDFMidpointJetProducer.cc,v 1.1 2007/06/30 17:24:06 fedor Exp $
//
//--------------------------------------------
#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CDFMidpointJetProducer.h"

using namespace std;
using namespace reco;

namespace cms {
  CDFMidpointJetProducer::CDFMidpointJetProducer(edm::ParameterSet const& conf):
    BaseJetProducer (conf),
    alg_(conf)
  {}
  
  
  // run algorithm itself
  bool CDFMidpointJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
					     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}

