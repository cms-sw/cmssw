// File: KtJetProducer.cc
// Description:  see KtJetProducer.h
// Author:  Fernando Varela Rodriguez, Boston University
// Creation Date:  Apr. 22 2005 Initial version.
// Revisions:  R. Harris, 19-Oct-2005, modified to use real CaloTowers from Jeremy Mans
// Revisions:  F.Ratnikov, 8-Mar-2006, accommodate Candidate model
// $Id: KtJetProducer.cc,v 1.11 2006/03/31 20:57:52 fedor Exp $
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

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  KtJetProducer::KtJetProducer(const edm::ParameterSet& conf)
  : alg_(conf.getParameter<int>("ktAngle"),
	 conf.getParameter<int>("ktRecom"),
	 conf.getParameter<double>("ktECut"),
	 conf.getParameter<double>("ktRParam")),
    src_(conf.getParameter<string>( "src" ))
  {
    produces<CaloJetCollection>();
    produces<GenJetCollection>();
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
    auto_ptr<CaloJetCollection> caloJets(new CaloJetCollection);  //Empty Jet Coll
    auto_ptr<GenJetCollection> genJets(new GenJetCollection);  //Empty Jet Coll
    vector <ProtoJet>::const_iterator protojet = output.begin ();
    JetMaker jetMaker;
    for (; protojet != output.end (); protojet++) {
      if (jetMaker.convertableToCaloJet (*protojet)) {
	caloJets->push_back (jetMaker.makeCaloJet (*protojet));
      }
      if (jetMaker.convertableToGenJet (*protojet)) { 
	genJets->push_back (jetMaker.makeGenJet (*protojet));
      }
    }
    // store output
    if (!caloJets->empty ()) e.put(caloJets);  //Puts Jet Collection into event
    if (!genJets->empty ()) e.put(genJets);  //Puts Jet Collection into event
  }
}
