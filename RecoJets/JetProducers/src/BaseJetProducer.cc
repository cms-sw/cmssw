// File: BaseJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BaseJetProducer.cc,v 1.5 2006/12/05 18:37:46 fedor Exp $
//--------------------------------------------
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
using namespace JetReco;

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

  template <class T>  
  void dumpJets (const T& fJets) {
    for (unsigned i = 0; i < fJets.size(); ++i) {
      std::cout << "CaloJet # " << i << std::endl << fJets[i].print();
    }
  }
}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  BaseJetProducer::BaseJetProducer(const edm::ParameterSet& conf)
    : src_(conf.getParameter<edm::InputTag>( "src" )),
      jetType_ (conf.getUntrackedParameter<string>( "jetType", "CaloJet")),
      verbose_ (conf.getUntrackedParameter<bool>("verbose", false))
  {
    std::string alias = conf.getUntrackedParameter<string>( "alias", conf.getParameter<std::string>("@module_label"));
    if (makeCaloJet (jetType_)) produces<CaloJetCollection>().setBranchAlias (alias);
    else if (makeGenJet (jetType_)) produces<GenJetCollection>().setBranchAlias (alias);
    else if (makeBasicJet (jetType_)) produces<BasicJetCollection>().setBranchAlias (alias);
  }

  // Virtual destructor needed.
  BaseJetProducer::~BaseJetProducer() { } 

  // Functions that gets called by framework every event
  void BaseJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // get input
    edm::Handle<CandidateCollection> inputs;
    e.getByLabel( src_, inputs );                    
    InputCollection input;
    vector <ProtoJet> output;
    // fill input
    input.reserve (inputs->size ());
    if (verbose_) {
      std::cout << "BaseJetProducer::produce-> INPUT COLLECTION " << src_ << std::endl;
      for (unsigned i = 0; i < inputs->size (); i++) {
	std::cout << "  Constituent " << i << ", px/py/pz/pt/e: "
		  << (*inputs)[i].px() << '/' << (*inputs)[i].py() << '/' << (*inputs)[i].pz() << '/' 
		  << (*inputs)[i].pt() << '/' << (*inputs)[i].energy()
 		  << std::endl;
      }
    }
    for (unsigned i = 0; i < inputs->size (); i++) {
      input.push_back (InputItem (inputs, i));
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
    if (verbose_) {
      std::cout << "OUTPUT JET COLLECTION:" << std::endl;
    }
    if (caloJets.get ()) {
      GreaterByPt<CaloJet> compJets;
      std::sort (caloJets->begin (), caloJets->end (), compJets);
      if (verbose_) dumpJets (*caloJets);
      e.put(caloJets);  //Puts Jet Collection into event
    }
    if (genJets.get ()) {
      GreaterByPt<GenJet> compJets;
      std::sort (genJets->begin (), genJets->end (), compJets);
      if (verbose_) dumpJets (*caloJets);
      e.put(genJets);  //Puts Jet Collection into event
    }
    if (basicJets.get ()) {
      GreaterByPt<BasicJet> compJets;
      std::sort (basicJets->begin (), basicJets->end (), compJets);
      if (verbose_) dumpJets (*caloJets);
      e.put(basicJets);  //Puts Jet Collection into event
    }
    // output printout
    
  }
}
