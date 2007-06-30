// File: MidpointPilupSubtractionJetProducer.cc
// Description:  see MidpointPilupSubtractionJetProducer.h
// Author:  M. Paterno
// Creation Date:  MFP Apr. 6 2005 Initial version.
// Revision:  R. Harris,  Oct. 19, 2005 Modified to use real CaloTowers from Jeremy Mans
// Revisions:  F.Ratnikov, 8-Mar-2006, accommodate Candidate model
// $Id: MidpointPilupSubtractionJetProducer.cc,v 1.1 2007/05/04 09:53:18 kodolova Exp $
//
//--------------------------------------------
#include <memory>

#include "MidpointPilupSubtractionJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.

  MidpointPilupSubtractionJetProducer::MidpointPilupSubtractionJetProducer(edm::ParameterSet const& conf):
    BasePilupSubtractionJetProducer (conf),
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("coneRadius"),
	 conf.getParameter<double>("coneAreaFraction"),
	 conf.getParameter<int>("maxPairSize"),
	 conf.getParameter<int>("maxIterations"),
	 conf.getParameter<double>("overlapThreshold"),
	 conf.getUntrackedParameter<int>("debugLevel",0))
  {}


  // run algorithm itself
  bool MidpointPilupSubtractionJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}
