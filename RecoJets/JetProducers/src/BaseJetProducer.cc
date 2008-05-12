// File: BaseJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BaseJetProducer.cc,v 1.30 2007/10/17 21:38:01 fedor Exp $
//--------------------------------------------
#include <memory>

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "BaseJetProducer.h"

using namespace std;
using namespace reco;
using namespace JetReco;

namespace {
  const bool debug = false;

  bool makeCaloJet (const string& fTag) {
    return fTag == "CaloJet";
  }
  bool makePFJet (const string& fTag) {
    return fTag == "PFJet";
  }
  bool makeGenJet (const string& fTag) {
    return fTag == "GenJet";
  }
  bool makeBasicJet (const string& fTag) {
    return fTag == "BasicJet";
  }

  bool makeGenericJet (const string& fTag) {
    return !makeCaloJet (fTag) && !makePFJet (fTag) && !makeGenJet (fTag) && !makeBasicJet (fTag);
  }

  template <class T>  
  void dumpJets (const T& fJets) {
    for (unsigned i = 0; i < fJets.size(); ++i) {
      std::cout << "Jet # " << i << std::endl << fJets[i].print();
    }
  }

  void copyVariables (const ProtoJet& fProtojet, reco::Jet* fJet) {
    fJet->setJetArea (fProtojet.jetArea ());
    fJet->setPileup (fProtojet.pileup ());
    fJet->setNPasses (fProtojet.nPasses ());
  }

  void copyConstituents (const JetReco::InputCollection& fConstituents, const edm::View <Candidate>& fInput, reco::Jet* fJet) {
    // put constituents
    for (unsigned iConstituent = 0; iConstituent < fConstituents.size (); ++iConstituent) {
      fJet->addDaughter (fInput.ptrAt (fConstituents[iConstituent].index ()));
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
    if (makeCaloJet (mJetType)) {
      produces<CaloJetCollection>().setBranchAlias (alias);
    }
    else if (makePFJet (mJetType)) produces<PFJetCollection>().setBranchAlias (alias);
    else if (makeGenJet (mJetType)) produces<GenJetCollection>().setBranchAlias (alias);
    else if (makeBasicJet (mJetType)) produces<BasicJetCollection>().setBranchAlias (alias);
//     else if (makeGenericJet (mJetType)) produces<GenericJetCollection>().setBranchAlias (alias);
  }

  // Virtual destructor needed.
  BaseJetProducer::~BaseJetProducer() { } 

  // Functions that gets called by framework every event
  void BaseJetProducer::produce(edm::Event& e, const edm::EventSetup& fSetup)
  {
    // get input
    edm::Handle<edm::View <Candidate> > inputHandle; 
    e.getByLabel( mSrc, inputHandle);
    // convert to input collection
    JetReco::InputCollection input;
    input.reserve (inputHandle->size());
    for (unsigned i = 0; i < inputHandle->size(); ++i) {
      if ((mEtInputCut <= 0 || (*inputHandle)[i].et() > mEtInputCut) &&
	  (mEInputCut <= 0 || (*inputHandle)[i].energy() > mEInputCut)) {
	input.push_back (JetReco::InputItem (&((*inputHandle)[i]), i));
      }
    }
    if (mVerbose) {
      std::cout << "BaseJetProducer::produce-> INPUT COLLECTION selected from" << mSrc 
		<< " with ET > " << mEtInputCut << " and/or E > " << mEInputCut << std::endl;
      for (unsigned index = 0; index < input.size(); ++index) {
	std::cout << "  Input " << index << ", px/py/pz/pt/e: "
		  << input[index]->px() << '/' << input[index]->py() << '/' << input[index]->pz() << '/' 
		  << input[index]->pt() << '/' << input[index]->energy()
 		  << std::endl;
      }
    }
    
    // run algorithm
    vector <ProtoJet> output;
    if (input.empty ()) {
      edm::LogInfo ("Empty Event") << "empty input for jet algorithm: bypassing..." << std::endl;
    }
    else {
      runAlgorithm (input, &output);
    }

    // produce output collection
    if (mVerbose) {
      std::cout << "OUTPUT JET COLLECTION:" << std::endl;
    }
    reco::Jet::Point vertex (0,0,0); // do not have true vertex yet, use default
    // make sure protojets are sorted
    sortByPt (&output);
    if (makeCaloJet (mJetType)) {
      edm::ESHandle<CaloGeometry> geometry;
      fSetup.get<IdealGeometryRecord>().get(geometry);
      const CaloSubdetectorGeometry* towerGeometry = 
	geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);
      auto_ptr<CaloJetCollection> jets (new CaloJetCollection);
      for (unsigned iJet = 0; iJet < output.size (); ++iJet) {
	ProtoJet* protojet = &(output [iJet]);
	const JetReco::InputCollection& constituents = protojet->getTowerList();
	CaloJet::Specific specific;
	JetMaker::makeSpecific (constituents, *towerGeometry, &specific);
	jets->push_back (CaloJet (protojet->p4(), vertex, specific));
	Jet* newJet = &(jets->back());
	copyConstituents (constituents, *inputHandle, newJet);
	copyVariables (*protojet, newJet);
      }
      if (mVerbose) dumpJets (*jets);
      e.put(jets);
    }
    else if (makePFJet (mJetType)) {
      auto_ptr<PFJetCollection> jets (new PFJetCollection);
      for (unsigned iJet = 0; iJet < output.size (); ++iJet) {
	ProtoJet* protojet = &(output [iJet]);
	const JetReco::InputCollection& constituents = protojet->getTowerList();
	PFJet::Specific specific;
	JetMaker::makeSpecific (constituents, &specific);
	jets->push_back (PFJet (protojet->p4(), vertex, specific));
	Jet* newJet = &(jets->back());
	copyConstituents (constituents, *inputHandle, newJet);
	copyVariables (*protojet, newJet);
      }
      if (mVerbose) dumpJets (*jets);
      e.put(jets);
    }
    else if (makeGenJet (mJetType)) {
      auto_ptr<GenJetCollection> jets (new GenJetCollection);
      for (unsigned iJet = 0; iJet < output.size (); ++iJet) {
	ProtoJet* protojet = &(output [iJet]);
	const JetReco::InputCollection& constituents = protojet->getTowerList();
	GenJet::Specific specific;
	JetMaker::makeSpecific (constituents, &specific);
	jets->push_back (GenJet (protojet->p4(), vertex, specific));
	Jet* newJet = &(jets->back());
	copyConstituents (constituents, *inputHandle, newJet);
	copyVariables (*protojet, newJet);
      }
      if (mVerbose) dumpJets (*jets);
      e.put(jets);
    }
    else if (makeBasicJet (mJetType)) {
      auto_ptr<BasicJetCollection> jets (new BasicJetCollection);
      for (unsigned iJet = 0; iJet < output.size (); ++iJet) {
	ProtoJet* protojet = &(output [iJet]);
	const JetReco::InputCollection& constituents = protojet->getTowerList();
	jets->push_back (BasicJet (protojet->p4(), vertex));
	Jet* newJet = &(jets->back());
	copyConstituents (constituents, *inputHandle, newJet);
	copyVariables (*protojet, newJet);
      }
      if (mVerbose) dumpJets (*jets);
      e.put(jets);
    }
//     else if (makeGenericJet (mJetType)) {
//       auto_ptr<GenericJetCollection> jets (new GenericJetCollection);
//       for (unsigned iJet = 0; iJet < output.size (); ++iJet) {
// 	ProtoJet* protojet = output [iJet];
// 	const JetReco::InputCollection& constituents = protojet->getTowerList();
// 	jets->push_back (GenericJet (protojet->p4()));
// 	Jet* newJet = &(jets->back());
// 	copyConstituents (constituents, *inputHandle, newJet);
// 	copyVariables (*protojet, newJet);
//       }
//       if (mVerbose) dumpJets (*jets);
//       e.put(jets);
//     }
  }
}

