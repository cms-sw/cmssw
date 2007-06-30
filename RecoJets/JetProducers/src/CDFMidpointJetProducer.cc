// File: CDFMidpointJetProducer.cc
// Description:  see CDFMidpointJetProducer.h
// Author:  M. Fedor Ratnikov, Maryland
// $Id: CDFMidpointJetProducer.cc,v 1.22 2007/05/19 04:20:10 fedor Exp $
//
//--------------------------------------------
#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CDFMidpointJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.

  CDFMidpointJetProducer::CDFMidpointJetProducer(edm::ParameterSet const& conf):
    BaseJetProducer (conf),
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("coneRadius"),
	 conf.getParameter<double>("coneAreaFraction"),
	 conf.getParameter<int>("maxPairSize"),
	 conf.getParameter<int>("maxIterations"),
	 conf.getParameter<double>("overlapThreshold"),
	 conf.getUntrackedParameter<int>("debugLevel",0))
  {}


  // run algorithm itself
  bool CDFMidpointJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}

