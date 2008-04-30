/*
 *  See header file for a description of this class.
 *
 *  $Date:$
 *  $Revision:$
 *  \author F. Chlebana - Fermilab
 */

#include "DQMOffline/JetMET/src/JetAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;

JetAnalyzer::JetAnalyzer(const edm::ParameterSet& pSet) {

  cout<<"[JetAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}

JetAnalyzer::~JetAnalyzer() { }

void JetAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "jetAnalyzer";

  LogTrace(metname)<<"[JetAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/JetAnalyzer");

  jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
  jetME->setBinLabel(1,"CaloJets",1);

  // monitoring of eta parameter
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");

  // monitoring of phi paramater
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");

  // monitoring of the transverse momentum
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  mEta                    = dbe->book1D("Eta", "Eta", etaBin, etaMin, etaMax);
  mPhi                    = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5);
  mE                      = dbe->book1D("E", "E", 100, 0, 500);
  mP                      = dbe->book1D("P", "P", 100, 0, 500);
  mPt                     = dbe->book1D("Pt", "Pt", 100, 0, 50);
  mPt_1                   = dbe->book1D("Pt1", "Pt1", 100, 0, 100);
  mPt_2                   = dbe->book1D("Pt2", "Pt2", 100, 0, 300);
  mPt_3                   = dbe->book1D("Pt3", "Pt3", 100, 0, 5000);
  mMass                   = dbe->book1D("Mass", "Mass", 100, 0, 25);
  mConstituents           = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100);
  //
  mEtaFirst               = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5);
  mPhiFirst               = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
  mEFirst                 = dbe->book1D("EFirst", "EFirst", 100, 0, 1000);
  mPtFirst                = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500);
  //
  mMaxEInEmTowers         = dbe->book1D("MaxEInEmTowers", "MaxEInEmTowers", 100, 0, 100);
  mMaxEInHadTowers        = dbe->book1D("MaxEInHadTowers", "MaxEInHadTowers", 100, 0, 100);
  mHadEnergyInHO          = dbe->book1D("HadEnergyInHO", "HadEnergyInHO", 100, 0, 10);
  mHadEnergyInHB          = dbe->book1D("HadEnergyInHB", "HadEnergyInHB", 100, 0, 50);
  mHadEnergyInHF          = dbe->book1D("HadEnergyInHF", "HadEnergyInHF", 100, 0, 50);
  mHadEnergyInHE          = dbe->book1D("HadEnergyInHE", "HadEnergyInHE", 100, 0, 100);
  mEmEnergyInEB           = dbe->book1D("EmEnergyInEB", "EmEnergyInEB", 100, 0, 50);
  mEmEnergyInEE           = dbe->book1D("EmEnergyInEE", "EmEnergyInEE", 100, 0, 50);
  mEmEnergyInHF           = dbe->book1D("EmEnergyInHF", "EmEnergyInHF", 120, -20, 100);
  mEnergyFractionHadronic = dbe->book1D("EnergyFractionHadronic", "EnergyFractionHadronic", 120, -0.1, 1.1);
  mEnergyFractionEm       = dbe->book1D("EnergyFractionEm", "EnergyFractionEm", 120, -0.1, 1.1);
  mN90                    = dbe->book1D("N90", "N90", 50, 0, 50);

}

void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::CaloJet& jet) {

  LogTrace(metname)<<"[JetAnalyzer] Analyze Calo Jet";

  jetME->Fill(1);

  if (mEta) mEta->Fill (jet.eta());
  if (mPhi) mPhi->Fill (jet.phi());
  if (mE) mE->Fill (jet.energy());
  if (mP) mP->Fill (jet.p());
  if (mPt) mPt->Fill (jet.pt());
  if (mPt_1) mPt_1->Fill (jet.pt());
  if (mPt_2) mPt_2->Fill (jet.pt());
  if (mPt_3) mPt_3->Fill (jet.pt());
  if (mMass) mMass->Fill (jet.mass());
  if (mConstituents) mConstituents->Fill (jet.nConstituents());
  //  if (jet == jet.begin ()) { // first jet
  //    if (mEtaFirst) mEtaFirst->Fill (jet.eta());
  //    if (mPhiFirst) mPhiFirst->Fill (jet.phi());
  //    if (mEFirst) mEFirst->Fill (jet.energy());
  //    if (mPtFirst) mPtFirst->Fill (jet.pt());
  //  }
  if (mMaxEInEmTowers) mMaxEInEmTowers->Fill (jet.maxEInEmTowers());
  if (mMaxEInHadTowers) mMaxEInHadTowers->Fill (jet.maxEInHadTowers());
  if (mHadEnergyInHO) mHadEnergyInHO->Fill (jet.hadEnergyInHO());
  if (mHadEnergyInHB) mHadEnergyInHB->Fill (jet.hadEnergyInHB());
  if (mHadEnergyInHF) mHadEnergyInHF->Fill (jet.hadEnergyInHF());
  if (mHadEnergyInHE) mHadEnergyInHE->Fill (jet.hadEnergyInHE());
  if (mEmEnergyInEB) mEmEnergyInEB->Fill (jet.emEnergyInEB());
  if (mEmEnergyInEE) mEmEnergyInEE->Fill (jet.emEnergyInEE());
  if (mEmEnergyInHF) mEmEnergyInHF->Fill (jet.emEnergyInHF());
  if (mEnergyFractionHadronic) mEnergyFractionHadronic->Fill (jet.energyFractionHadronic());
  if (mEnergyFractionEm) mEnergyFractionEm->Fill (jet.emEnergyFraction());
  if (mN90) mN90->Fill (jet.n90());  

}
