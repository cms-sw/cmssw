// File: KtJetProducer.cc
// Description:  see KtJetProducer.h
// Author:  Fernando Varela Rodriguez, Boston University
// Creation Date:  Apr. 22 2005 Initial version.
// Revisions:  R. Harris, 19-Oct-2005, modified to use real CaloTowers from Jeremy Mans
// Revisions:  F.Ratnikov, 8-Mar-2006, accommodate Candidate model
// $Id: KtJetProducer.cc,v 1.12 2006/04/05 00:26:57 fedor Exp $
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
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
  KtJetProducer::KtJetProducer(const edm::ParameterSet& conf)
  : alg_(conf.getParameter<int>("ktAngle"),
	 conf.getParameter<int>("ktRecom"),
	 conf.getParameter<double>("ktECut"),
	 conf.getParameter<double>("ktRParam")),
    src_(conf.getParameter<string>( "src" )),
    jetType_ (conf.getUntrackedParameter<string>( "jetType", "CaloJet"))
  {
    if (makeCaloJet (jetType_)) produces<CaloJetCollection>();
    if (makeGenJet (jetType_)) produces<GenJetCollection>();
  }

  // Virtual destructor needed.
  KtJetProducer::~KtJetProducer() { }  

  // Functions that gets called by framework every event
  void KtJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // get input
    edm::Handle<CandidateCollection> inputs;
    e.getByLabel( src_, inputs );                    
    vector <const Candidate*> input;
    vector <ProtoJet> output;
    // fill input
    input.reserve (inputs->size ());
    CandidateCollection::const_iterator input_object = inputs->begin ();
    for (; input_object != inputs->end (); input_object++) {
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
      }
      if (genJets.get ()) { 
	genJets->push_back (jetMaker.makeGenJet (*protojet));
      }
    }
    // store output
    if (caloJets.get ()) e.put(caloJets);  //Puts Jet Collection into event
    if (genJets.get ()) e.put(genJets);  //Puts Jet Collection into event
  }
}
