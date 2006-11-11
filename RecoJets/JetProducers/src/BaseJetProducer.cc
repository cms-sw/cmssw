// File: BaseJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BaseJetProducer.cc,v 1.2 2006/09/08 23:34:35 fedor Exp $
//--------------------------------------------
//test
#include <memory>

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"

#include "RecoJets/JetProducers/interface/BaseJetProducer.h"

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
  bool makeBasicJet (const string& fTag) {
    return fTag == "BasicJet";
  }
}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  BaseJetProducer::BaseJetProducer(const edm::ParameterSet& conf)
    : src_(conf.getParameter<edm::InputTag>( "src" )),
      jetType_ (conf.getUntrackedParameter<string>( "jetType", "CaloJet"))
  {}

  // Virtual destructor needed.
  BaseJetProducer::~BaseJetProducer() { } 

  //init branches and set alias name 
  void BaseJetProducer::initBranch (const std::string& fName) {
    if (makeCaloJet (jetType_)) produces<CaloJetCollection>().setBranchAlias (fName);
    if (makeGenJet (jetType_)) produces<GenJetCollection>().setBranchAlias (fName);
    if (makeBasicJet (jetType_)) produces<BasicJetCollection>().setBranchAlias (fName);
  }

  // Functions that gets called by framework every event
  void BaseJetProducer::produce(edm::Event& e, const edm::EventSetup&)
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
    runAlgorithm (input, &output);
    // produce output collection
    auto_ptr<CaloJetCollection> caloJets;
    if (makeCaloJet (jetType_)) caloJets.reset (new CaloJetCollection);
    auto_ptr<GenJetCollection> genJets;
    if (makeGenJet (jetType_)) genJets.reset (new GenJetCollection);
    auto_ptr<BasicJetCollection> basicJets;
    if (makeBasicJet (jetType_)) basicJets.reset (new BasicJetCollection);
    vector <ProtoJet>::const_iterator protojet = output.begin ();
    JetMaker jetMaker;
    for (; protojet != output.end (); protojet++) {
      if (caloJets.get ()) {
	caloJets->push_back (jetMaker.makeCaloJet (*protojet));
      }
      if (genJets.get ()) { 
	genJets->push_back (jetMaker.makeGenJet (*protojet));
      }
      if (basicJets.get ()) { 
	basicJets->push_back (jetMaker.makeBasicJet (*protojet));
      }
    }
    // sort and store output
    if (caloJets.get ()) {
      GreaterByPt<CaloJet> compJets;
      std::sort (caloJets->begin (), caloJets->end (), compJets);
      e.put(caloJets);  //Puts Jet Collection into event
    }
    if (genJets.get ()) {
      GreaterByPt<GenJet> compJets;
      std::sort (genJets->begin (), genJets->end (), compJets);
      e.put(genJets);  //Puts Jet Collection into event
    }
    if (basicJets.get ()) {
      GreaterByPt<BasicJet> compJets;
      std::sort (basicJets->begin (), basicJets->end (), compJets);
      e.put(basicJets);  //Puts Jet Collection into event
    }
  }
}
