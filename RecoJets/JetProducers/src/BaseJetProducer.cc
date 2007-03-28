// File: BaseJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BaseJetProducer.cc,v 1.11 2007/03/26 22:05:40 fedor Exp $
//--------------------------------------------
#include <memory>

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/GenericJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

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

  bool makeGenericJet (const string& fTag) {
    return !makeCaloJet (fTag) && !makeGenJet (fTag) && !makeBasicJet (fTag);
  }

  template <class T>  
  void dumpJets (const T& fJets) {
    for (unsigned i = 0; i < fJets.size(); ++i) {
      std::cout << "Jet # " << i << std::endl << fJets[i].print();
    }
  }

  class FakeHandle {
  public:
    FakeHandle (const CandidateCollection* fCollection, edm::ProductID fId) : mCollection (fCollection), mId (fId) {}
    edm::ProductID id () const {return mId;} 
    const CandidateCollection* product () const {return mCollection;}
  private:
    const CandidateCollection* mCollection;
    edm::ProductID mId;
  };

  class FakeCandidate : public RecoCandidate {
  public:
     FakeCandidate( Charge q , const LorentzVector& p4, const Point& vtx) : RecoCandidate( q, p4, vtx ) {}
  private:
    virtual bool overlap( const Candidate & ) const {return false;}
  };
  
  template <class HandleC>
  void fillInputs (const HandleC& fData, JetReco::InputCollection* fInput, double fEtCut, double fECut) {
    for (unsigned i = 0; i < fData.product ()->size (); i++) {
      const reco::Candidate* constituent = &((*(fData.product ()))[i]);
      if ((fEtCut <= 0 || constituent->et() > fEtCut) &&
	  (fECut <= 0 || constituent->energy() > fECut)) {
	fInput->push_back (InputItem (fData, i));
      }
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
    else if (makeGenericJet (mJetType)) produces<GenericJetCollection>().setBranchAlias (alias);
  }

  // Virtual destructor needed.
  BaseJetProducer::~BaseJetProducer() { } 

  // Functions that gets called by framework every event
  void BaseJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // get input
    InputCollection input;
    CandidateCollection inputCache;
    if (makeGenericJet (mJetType)) {
      edm::Handle<edm::View <Candidate> > genericInputs; 
      e.getByLabel( mSrc, genericInputs ); 
      for (unsigned i = 0; i < genericInputs->size (); ++i) {
	const Candidate* ref = &((*genericInputs)[i]);
	Candidate* c = new FakeCandidate (ref->charge (), ref->p4 (), ref->vertex ());
	inputCache.push_back (c);
      }
      FakeHandle handle (&inputCache, genericInputs.id ());
      fillInputs (handle, &input, mEtInputCut, mEInputCut);
    }
    else { // CandidateCollection
      edm::Handle<CandidateCollection> concreteInputs;
      e.getByLabel( mSrc, concreteInputs );
      fillInputs (concreteInputs, &input, mEtInputCut, mEInputCut);
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
    vector <ProtoJet> output;
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
    auto_ptr<GenericJetCollection> genericJets;
    if (makeGenericJet (mJetType)) genericJets.reset (new GenericJetCollection);
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
      if (genericJets.get ()) { 
	genericJets->push_back (jetMaker.makeGenericJet (*protojet));
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
    if (genericJets.get ()) {
      GreaterByPt<GenericJet> compJets;
      std::sort (genericJets->begin (), genericJets->end (), compJets);
      if (mVerbose) dumpJets (*genericJets);
      e.put(genericJets);  //Puts Jet Collection into event
    }
  }
}

