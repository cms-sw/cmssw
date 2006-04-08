// File: MidpointJetProducer.cc
// Description:  see MidpointJetProducer.h
// Author:  M. Paterno
// Creation Date:  MFP Apr. 6 2005 Initial version.
// Revision:  R. Harris,  Oct. 19, 2005 Modified to use real CaloTowers from Jeremy Mans
// Revisions:  F.Ratnikov, 8-Mar-2006, accommodate Candidate model
// $Id: MidpointJetProducer.cc,v 1.11 2006/04/05 00:26:57 fedor Exp $
//
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

  bool makeCaloJet (const string& fTag) {
    return fTag == "CaloJet";
  }
  bool makeGenJet (const string& fTag) {
    return fTag == "GenJet";
  }
}

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
	 conf.getUntrackedParameter<int>("debugLevel",0)),
    src_(conf.getParameter<string>( "src" )),
    jetType_ (conf.getUntrackedParameter<string>( "jetType", "CaloJet"))
  {
    if (makeCaloJet (jetType_)) produces<CaloJetCollection>();
    if (makeGenJet (jetType_)) produces<GenJetCollection>();
  }

  // Virtual destructor needed.
  MidpointJetProducer::~MidpointJetProducer() { }  

  // Functions that gets called by framework every event
  void MidpointJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    if (debug) {
      std::cout << "MidpointJetProducer::produce->" 
	//	      << " run:" <<  e.getRun()
		<< " event:" << e.id ()
		<< " timestamp:" << e.time ().value ()
	//	      << " section:" << e.getLuminositySection()
		<< "\n ====================================================="
		<< std::endl;
    } 
    // get input
    edm::Handle<CandidateCollection> inputs;
    e.getByLabel( src_, inputs );                    
    vector <const Candidate*> input;
    vector <ProtoJet> output;
    // fill input
    input.reserve (inputs->size ());
    CandidateCollection::const_iterator input_object = inputs->begin ();
    for (; input_object != inputs->end (); input_object++) {
      if (debug) {
	std::cout << "MidpointJetProducer::produce-> input E/p/eta/phi/M: " << input_object->energy ()
		  << '/' << input_object->p() << '/' << input_object->eta ()
		  << '/' << input_object->phi() << '/' << input_object->mass ()
		  << std::endl;
      }
      input.push_back (&*input_object); 
    }
    // run algorithm
    alg_.run (input, &output);
    // produce output collection
    auto_ptr<CaloJetCollection> caloJets;
    if (makeCaloJet (jetType_)) caloJets.reset (new CaloJetCollection);
    auto_ptr<GenJetCollection> genJets;
    if (makeGenJet (jetType_)) genJets.reset (new GenJetCollection);

    vector <ProtoJet>::const_iterator protojet = output.begin ();
    JetMaker jetMaker;
    for (; protojet != output.end (); protojet++) {
      if (caloJets.get ()) {
	caloJets->push_back (jetMaker.makeCaloJet (*protojet));
	if (debug) std::cout << "MidpointJetProducer::produce-> add protojet to CaloJets." << std::endl;
      }
      if (genJets.get ()) { 
	genJets->push_back (jetMaker.makeGenJet (*protojet));
	if (debug) std::cout << "MidpointJetProducer::produce-> add protojet to GenJets." << std::endl;
      }
    }
    // store output
    if (caloJets.get ()) e.put(caloJets);  //Puts Jet Collection into event
    if (genJets.get ()) e.put(genJets);  //Puts Jet Collection into event
  }

}
