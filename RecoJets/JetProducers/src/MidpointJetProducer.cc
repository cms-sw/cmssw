// File: MidpointJetProducer.cc
// Description:  see MidpointJetProducer.h
// Author:  M. Paterno
// Creation Date:  MFP Apr. 6 2005 Initial version.
//
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"
#include "DataFormats/CaloObjects/interface/CaloTowerCollection.h"
#include "FWCore/CoreFramework/interface/Handle.h"

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
	 conf.getUntrackedParameter<int>("debugLevel",0))
  {
  }

  // Virtual destructor needed.
  MidpointJetProducer::~MidpointJetProducer() { }  

  // Functions that gets called by framework every event
  void MidpointJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // Step A: Get Inputs 
    edm::Handle<CaloTowerCollection> towers;  //Fancy Event Pointer to CaloTowers
    e.getByLabel("CalTwr", towers);           //Set pointer to CaloTowers

    // Step B: Create empty output 
    std::auto_ptr<CaloJetCollection> result(new CaloJetCollection);  //Empty Jet Coll

    // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
    alg_.run(towers.product(), *result);  //Makes Full Jet Collection

    // Step D: Put outputs into event
    e.put(result);  //Puts Jet Collection into event
  }

}
