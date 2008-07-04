/*
 *  See header file for a description of this class.
 *
 *  $Date:$
 *  $Revision:$
 *  \author F. Chlebana - Fermilab
 */

#include "DQMOffline/JetMET/src/PFJetAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

// #include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;


PFJetAnalyzer::PFJetAnalyzer(const edm::ParameterSet& pSet) {

  cout<<"[PFJetAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}


PFJetAnalyzer::~PFJetAnalyzer() { }


void PFJetAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "pFJetAnalyzer";

  LogTrace(metname)<<"[PFJetAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/PFJetAnalyzer");

  //  jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
  //  jetME->setBinLabel(2,"PFJets",1);

  mEta = dbe->book1D("Eta", "Eta", 100, -5, 5);
  mPhi = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5);
  mE = dbe->book1D("E", "E", 100, 0, 500);
  mP = dbe->book1D("P", "P", 100, 0, 500);
  mPt = dbe->book1D("Pt", "Pt", 100, 0, 50);
  mMass = dbe->book1D("Mass", "Mass", 100, 0, 25);
  mConstituents = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100);
  //
  mEtaFirst = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5);
  mPhiFirst = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
  mEFirst = dbe->book1D("EFirst", "EFirst", 100, 0, 1000);
  mPtFirst = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500);
  //
  mChargedHadronEnergy = dbe->book1D("mChargedHadronEnergy", "mChargedHadronEnergy", 100, 0, 100);
  mNeutralHadronEnergy = dbe->book1D("mNeutralHadronEnergy", "mNeutralHadronEnergy", 100, 0, 100);
  mChargedEmEnergy= dbe->book1D("mChargedEmEnergy ", "mChargedEmEnergy ", 100, 0, 100);
  mChargedMuEnergy = dbe->book1D("mChargedMuEnergy", "mChargedMuEnergy", 100, 0, 100);
  mNeutralEmEnergy= dbe->book1D("mNeutralEmEnergy", "mNeutralEmEnergy", 100, 0, 100);
  mChargedMultiplicity= dbe->book1D("mChargedMultiplicity ", "mChargedMultiplicity ", 100, 0, 100);
  mNeutralMultiplicity = dbe->book1D(" mNeutralMultiplicity", "mNeutralMultiplicity", 100, 0, 100);
  mMuonMultiplicity= dbe->book1D("mMuonMultiplicity", "mMuonMultiplicity", 100, 0, 100);
  
  //__________________________________________________

  mNeutralFraction = dbe->book1D("NeutralFraction","Neutral Fraction",100,0,1);
  
}

void PFJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::PFJet& jet) {

  LogTrace(metname)<<"[JetAnalyzer] Analyze PFJet";

  //  jetME->Fill(2);

  if (mEta) mEta->Fill (jet.eta());
  if (mPhi) mPhi->Fill (jet.phi());
  if (mE) mE->Fill (jet.energy());
  if (mP) mP->Fill (jet.p());
  if (mPt) mPt->Fill (jet.pt());
  if (mMass) mMass->Fill (jet.mass());
  if (mConstituents) mConstituents->Fill (jet.nConstituents());
  //    if (jet == pfJets->begin ()) { // first jet
  //      if (mEtaFirst) mEtaFirst->Fill (jet.eta());
  //      if (mPhiFirst) mPhiFirst->Fill (jet.phi());
  //      if (mEFirst) mEFirst->Fill (jet.energy());
  //      if (mPtFirst) mPtFirst->Fill (jet.pt());
  //    }
  if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill (jet.chargedHadronEnergy());
  if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill (jet.neutralHadronEnergy());
  if (mChargedEmEnergy) mChargedEmEnergy->Fill(jet.chargedEmEnergy());
  if (mChargedMuEnergy) mChargedMuEnergy->Fill (jet.chargedMuEnergy ());
  if (mNeutralEmEnergy) mNeutralEmEnergy->Fill(jet.neutralEmEnergy());
  if (mChargedMultiplicity ) mChargedMultiplicity->Fill(jet.chargedMultiplicity());
  if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill(jet.neutralMultiplicity());
  if (mMuonMultiplicity )mMuonMultiplicity->Fill (jet. muonMultiplicity());
  //_______________________________________________________
  if (mNeutralFraction) mNeutralFraction->Fill (jet.neutralMultiplicity()/jet.nConstituents());

}
