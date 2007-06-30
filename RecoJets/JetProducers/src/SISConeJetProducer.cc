// File: SISConeJetProducer.cc
// Description:  see SISConeJetProducer.h
// Author:  Fedor Ratnikov, Maryland, June 30, 2007
// $Id: SISConeJetProducer.cc,v 1.22 2007/05/19 04:20:10 fedor Exp $
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
    alg_(
	 conf.getParameter<double>("coneRadius"),
	 conf.getParameter<double>("coneOverlapThreshold"),
	 conf.getParameter<std::string>("splitMergeScale"),
	 conf.getParameter<int>("maxPasses"),
	 conf.getParameter<double>("protojetPtMin"),
	 conf.getParameter<bool>("caching"),
	 conf.getUntrackedParameter<int>("debugLevel",0))
  {}


  // run algorithm itself
  bool SISConeJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);
    return true;
  }
}

