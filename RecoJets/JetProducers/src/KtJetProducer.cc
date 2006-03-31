// File: KtJetProducer.cc
// Description:  see KtJetProducer.h
// Author:  Fernando Varela Rodriguez, Boston University
// Creation Date:  Apr. 22 2005 Initial version.
// Revisions:  R. Harris, 19-Oct-2005, modified to use real CaloTowers from Jeremy Mans
// Revisions:  F.Ratnikov, 8-Mar-2006, accommodate Candidate model
// $Id: KtJetProducer.cc,v 1.10 2006/03/08 20:34:19 fedor Exp $
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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
  }

  // Virtual destructor needed.
  KtJetProducer::~KtJetProducer() { }  

  // Functions that gets called by framework every event
  void KtJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // get input
    edm::Handle<CandidateCollection> towers;
    e.getByLabel( src_, towers );                    
    vector <const Candidate*> input;
    vector <ProtoJet> output;
    // fill input
    input.reserve (towers->size ());
    CandidateCollection::const_iterator tower = towers->begin ();
    for (; tower != towers->end (); tower++) {
      input.push_back (&*tower); 
    }
    // run algorithm
    alg_.run (input, &output);
    // produce output collection
    auto_ptr<CaloJetCollection> result(new CaloJetCollection);  //Empty Jet Coll
    vector <ProtoJet>::const_iterator protojet = output.begin ();
    JetMaker jetMaker;
    for (; protojet != output.end (); protojet++) {
      result->push_back (jetMaker.makeCaloJet (*protojet));
    }
    // store output
    e.put(result);  //Puts Jet Collection into event
  }
}
