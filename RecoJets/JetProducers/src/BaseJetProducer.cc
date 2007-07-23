// File: BaseJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BaseJetProducer.cc,v 1.8 2007/02/08 21:07:04 fedor Exp $
//--------------------------------------------
#include <memory>

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
    : mSrc(conf.getParameter<edm::InputTag>( "src" )),
      mJetType (conf.getUntrackedParameter<string>( "jetType", "CaloJet")),
      mVerbose (conf.getUntrackedParameter<bool>("verbose", false)),
      mEtInputCut (conf.getParameter<double>("inputEtMin")),
      mEInputCut (conf.getParameter<double>("inputEMin"))
  {
    std::string alias = conf.getUntrackedParameter<string>( "alias", conf.getParameter<std::string>("@module_label"));
    if (makeCaloJet (mJetType)) produces<CaloJetCollection>().setBranchAlias (alias);
    else if (makeGenJet (mJetType)) produces<GenJetCollection>().setBranchAlias (alias);
    else if (makeBasicJet (mJetType)) produces<BasicJetCollection>().setBranchAlias (alias);
  }

  // Virtual destructor needed.
  BaseJetProducer::~BaseJetProducer() { } 

  // Functions that gets called by framework every event
  void BaseJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // get input
    edm::Handle<CandidateCollection> inputs;
    e.getByLabel( mSrc, inputs );                    
    InputCollection input;
    vector <ProtoJet> output;
    // fill input
    input.reserve (inputs->size ());
    for (unsigned i = 0; i < inputs->size (); i++) {
      const reco::Candidate* constituent = &(*inputs)[i];
      if ((mEtInputCut <= 0 || constituent->et() > mEtInputCut) &&
	  (mEInputCut <= 0 || constituent->energy() > mEInputCut)) {
	input.push_back (InputItem (inputs, i));
      }
    }
    if (mVerbose) {
      std::cout << "BaseJetProducer::produce-> INPUT COLLECTION selected from" << mSrc 
		<< " with ET > " << mEtInputCut << " and/or E > " << mEInputCut << std::endl;
      int index = 0;
      for (InputCollection::const_iterator constituent = input.begin();
       constituent != input.end(); ++constituent, ++index) {
	std::cout << "  Constituent " << index << ", px/py/pz/pt/e: "
		  << (*constituent)->px() << '/' << (*constituent)->py() << '/' << (*constituent)->pz() << '/' 
		  << (*constituent)->pt() << '/' << (*constituent)->energy()
 		  << std::endl;
      }
    }

    // run algorithm
    if (input.empty ()) {
      edm::LogWarning("Empty Event") << "empty input for jet algorithm: bypassing..." << std::endl;
    }
    else {
      runAlgorithm (input, &output);
    }

    // produce output collection
    auto_ptr<CaloJetCollection> caloJets;
    if (makeCaloJet (mJetType)) caloJets.reset (new CaloJetCollection);
    auto_ptr<GenJetCollection> genJets;
    if (makeGenJet (mJetType)) genJets.reset (new GenJetCollection);
    auto_ptr<BasicJetCollection> basicJets;
    if (makeBasicJet (mJetType)) basicJets.reset (new BasicJetCollection);
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
    if (mVerbose) {
      std::cout << "OUTPUT JET COLLECTION:" << std::endl;
    }
    if (caloJets.get ()) {
      GreaterByPt<CaloJet> compJets;
      std::sort (caloJets->begin (), caloJets->end (), compJets);
      if (mVerbose) dumpJets (*caloJets);
      e.put(caloJets);  //Puts Jet Collection into event
    }
    if (genJets.get ()) {
      GreaterByPt<GenJet> compJets;
      std::sort (genJets->begin (), genJets->end (), compJets);
      if (mVerbose) dumpJets (*genJets);
      e.put(genJets);  //Puts Jet Collection into event
    }
    if (basicJets.get ()) {
      GreaterByPt<BasicJet> compJets;
      std::sort (basicJets->begin (), basicJets->end (), compJets);
      if (mVerbose) dumpJets (*basicJets);
      e.put(basicJets);  //Puts Jet Collection into event
    }
    // output printout
    
  }
}
