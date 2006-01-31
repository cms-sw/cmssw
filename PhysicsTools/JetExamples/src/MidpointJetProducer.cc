// File: MidpointJetProducer.cc
// Description:  see MidpointJetProducer.h
// Author:  M. Paterno
// Creation Date:  MFP Apr. 6 2005 Initial version.
// Revision:  R. Harris,  Oct. 19, 2005 Modified to use real CaloTowers from Jeremy Mans
//
//--------------------------------------------
#include <memory>

#include "PhysicsTools/JetExamples/src/MidpointJetProducer.h"
#include "PhysicsTools/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace std;
using namespace aod;

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.

  MidpointJetProducer::MidpointJetProducer(edm::ParameterSet const& conf):
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("towerThreshold"),
	 conf.getParameter<double>("coneRadius"),
	 conf.getParameter<double>("coneAreaFraction"),
	 conf.getParameter<int>("maxPairSize"),
	 conf.getParameter<int>("maxIterations"),
	 conf.getParameter<double>("overlapThreshold"),
	 conf.getUntrackedParameter<int>("debugLevel",0) ),
    src_( conf.getParameter<string>( "src" ) ) {
    produces<CandidateCollection>();
  }

  MidpointJetProducer::~MidpointJetProducer() { }  

  void MidpointJetProducer::produce(edm::Event& e, const edm::EventSetup&) {
    edm::Handle<CandidateCollection> towers;
    e.getByLabel( src_, towers );                    
    std::auto_ptr<CandidateCollection> result(new CandidateCollection);
    alg_.run( towers.product(), *result );
    e.put( result );
  }

}
