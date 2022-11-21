// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/getRef.h"

#include "DataFormats/JetMatching/interface/JetFlavour.h"
#include "DataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "DataFormats/JetMatching/interface/MatchedPartons.h"
#include "DataFormats/JetMatching/interface/JetMatchedPartons.h"

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <Math/VectorUtil.h>
#include <TMath.h>

class printJetFlavour : public edm::one::EDAnalyzer<> {
public:
  explicit printJetFlavour(const edm::ParameterSet&);
  ~printJetFlavour(){};
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::InputTag sourcePartons_;
  edm::EDGetTokenT<reco::JetMatchedPartonsCollection> sourceByReferToken_;
  edm::EDGetTokenT<reco::JetFlavourMatchingCollection> sourceByValueToken_;
  edm::Handle<reco::JetMatchedPartonsCollection> theTagByRef;
  edm::Handle<reco::JetFlavourMatchingCollection> theTagByValue;
};

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

printJetFlavour::printJetFlavour(const edm::ParameterSet& iConfig) {
  sourceByReferToken_ = consumes<reco::JetMatchedPartonsCollection>(iConfig.getParameter<InputTag>("srcByReference"));
  sourceByValueToken_ = consumes<reco::JetFlavourMatchingCollection>(iConfig.getParameter<InputTag>("srcByValue"));
}

void printJetFlavour::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cout << "[printJetFlavour] analysing event " << iEvent.id() << endl;
  try {
    iEvent.getByToken(sourceByReferToken_, theTagByRef);
    iEvent.getByToken(sourceByValueToken_, theTagByValue);
  } catch (std::exception& ce) {
    cerr << "[printJetFlavour] caught std::exception " << ce.what() << endl;
    return;
  }

  cout << "-------------------- Jet Flavour by Ref From Partons--------------" << endl;
  for (JetMatchedPartonsCollection::const_iterator j = theTagByRef->begin(); j != theTagByRef->end(); j++) {
    const Jet* aJet = (*j).first.get();
    const MatchedPartons aMatch = (*j).second;
    printf("[printJetFlavour] (pt,eta,phi) jet = %7.2f %6.3f %6.3f \n", aJet->et(), aJet->eta(), aJet->phi());
    const GenParticleRef theHeaviest = aMatch.heaviest();
    if (theHeaviest.isNonnull()) {
      float dist = DeltaR(aJet->p4(), theHeaviest.get()->p4());
      cout << setprecision(2) << setw(6) << fixed
           << "                           theHeaviest flav (pt,eta,phi)=" << theHeaviest.get()->pdgId() << " ("
           << theHeaviest.get()->et() << "," << theHeaviest.get()->eta() << "," << theHeaviest.get()->phi()
           << ") Dr=" << dist << endl;
    }
    const GenParticleRef theNearest2 = aMatch.nearest_status2();
    if (theNearest2.isNonnull()) {
      float dist = DeltaR(aJet->p4(), theNearest2.get()->p4());
      cout << "                      theNearest Stat2 flav (pt,eta,phi)=" << theNearest2.get()->pdgId() << " ("
           << theNearest2.get()->et() << "," << theNearest2.get()->eta() << "," << theNearest2.get()->phi()
           << ") Dr=" << dist << endl;
    }
    const GenParticleRef theNearest3 = aMatch.nearest_status3();
    if (theNearest3.isNonnull()) {
      float dist = DeltaR(aJet->p4(), theNearest3.get()->p4());
      cout << "                      theNearest Stat3 flav (pt,eta,phi)=" << theNearest3.get()->pdgId() << " ("
           << theNearest3.get()->et() << "," << theNearest3.get()->eta() << "," << theNearest3.get()->phi()
           << ") Dr=" << dist << endl;
    }
    const GenParticleRef thePhyDef = aMatch.physicsDefinitionParton();
    if (thePhyDef.isNonnull()) {
      float dist = DeltaR(aJet->p4(), thePhyDef.get()->p4());
      cout << "                     thePhysDefinition flav (pt,eta,phi)=" << thePhyDef.get()->pdgId() << " ("
           << thePhyDef.get()->et() << "," << thePhyDef.get()->eta() << "," << thePhyDef.get()->phi() << ") Dr=" << dist
           << endl;
    }
    const GenParticleRef theAlgDef = aMatch.algoDefinitionParton();
    if (theAlgDef.isNonnull()) {
      float dist = DeltaR(aJet->p4(), theAlgDef.get()->p4());
      cout << "                     theAlgoDefinition flav (pt,eta,phi)=" << theAlgDef.get()->pdgId() << " ("
           << theAlgDef.get()->et() << "," << theAlgDef.get()->eta() << "," << theAlgDef.get()->phi() << ") Dr=" << dist
           << endl;
    }
  }

  cout << "-------------------- Jet Flavour by Value ------------------------" << endl;
  for (JetFlavourMatchingCollection::const_iterator j = theTagByValue->begin(); j != theTagByValue->end(); j++) {
    RefToBase<Jet> aJet = (*j).first;
    const JetFlavour aFlav = (*j).second;

    printf("[printJetFlavour] (pt,eta,phi) jet = %7.2f %6.3f %6.3f | parton = %7.2f %6.3f %6.3f | %4d\n",
           aJet.get()->et(),
           aJet.get()->eta(),
           aJet.get()->phi(),
           aFlav.getLorentzVector().pt(),
           aFlav.getLorentzVector().eta(),
           aFlav.getLorentzVector().phi(),
           aFlav.getFlavour());
  }
}

DEFINE_FWK_MODULE(printJetFlavour);
