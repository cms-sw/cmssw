/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/14 16:19:54 $
 *  $Revision: 1.6 $
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

// ***********************************************************
JetAnalyzer::JetAnalyzer(const edm::ParameterSet& pSet) {

  parameters   = pSet;
  _leadJetFlag = 0;
  _NJets       = 0;
  _JetLoPass   = 0;
  _JetHiPass   = 0;
}

// ***********************************************************
JetAnalyzer::~JetAnalyzer() { }


// ***********************************************************
void JetAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "jetAnalyzer";

  LogTrace(metname)<<"[JetAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/"+_source);

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

  // 
  eBin = parameters.getParameter<int>("eBin");
  eMin = parameters.getParameter<double>("eMin");
  eMax = parameters.getParameter<double>("eMax");

  // 
  pBin = parameters.getParameter<int>("pBin");
  pMin = parameters.getParameter<double>("pMin");
  pMax = parameters.getParameter<double>("pMax");


  // Generic Jet Parameters
  mEta                    = dbe->book1D("Eta", "Eta", etaBin, etaMin, etaMax);
  mPhi                    = dbe->book1D("Phi", "Phi", phiBin, phiMin, phiMax);
  mE                      = dbe->book1D("E", "E", eBin, eMin, eMax);
  mP                      = dbe->book1D("P", "P", pBin, pMin, pMax);
  mPt                     = dbe->book1D("Pt", "Pt", ptBin, ptMin, ptMax);
  mPt_1                   = dbe->book1D("Pt1", "Pt1", 100, 0, 100);
  mPt_2                   = dbe->book1D("Pt2", "Pt2", 100, 0, 300);
  mPt_3                   = dbe->book1D("Pt3", "Pt3", 100, 0, 5000);
  mMass                   = dbe->book1D("Mass", "Mass", 100, 0, 25);
  mConstituents           = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100);
  mNJets                  = dbe->book1D("NJets", "Number of Jets", 100, 0, 100);

  mPhi_Barrel             = dbe->book1D("Phi_Barrel", "Phi_Barrel", phiBin, phiMin, phiMax);
  mE_Barrel               = dbe->book1D("E_Barrel", "E_Barrel", eBin, eMin, eMax);
  mPt_Barrel              = dbe->book1D("Pt_Barrel", "Pt_Barrel", ptBin, ptMin, ptMax);

  mPhi_EndCap             = dbe->book1D("Phi_EndCap", "Phi_EndCap", phiBin, phiMin, phiMax);
  mE_EndCap               = dbe->book1D("E_EndCap", "E_EndCap", eBin, eMin, eMax);
  mPt_EndCap              = dbe->book1D("Pt_EndCap", "Pt_EndCap", ptBin, ptMin, ptMax);

  mPhi_Forward             = dbe->book1D("Phi_Forward", "Phi_Forward", phiBin, phiMin, phiMax);
  mE_Forward               = dbe->book1D("E_Forward", "E_Forward", eBin, eMin, eMax);
  mPt_Forward              = dbe->book1D("Pt_Forward", "Pt_Forward", ptBin, ptMin, ptMax);

  // Leading Jet Parameters
  mEtaFirst               = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5);
  mPhiFirst               = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
  mEFirst                 = dbe->book1D("EFirst", "EFirst", 100, 0, 1000);
  mPtFirst                = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500);

  // CaloJet specific
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

  // Low and high pt trigger paths
  mEta_Lo                 = dbe->book1D("Eta_Lo", "Eta Lo", etaBin, etaMin, etaMax);
  mPhi_Lo                 = dbe->book1D("Phi_Lo", "Phi Lo", phiBin, phiMin, phiMax);
  mPt_Lo                  = dbe->book1D("Pt_Lo", "Pt Lo", ptBin, ptMin, ptMax);

  mEta_Hi                 = dbe->book1D("Eta_Hi", "Eta Hi", etaBin, etaMin, etaMax);
  mPhi_Hi                 = dbe->book1D("Phi_Hi", "Phi Hi", phiBin, phiMin, phiMax);
  mPt_Hi                  = dbe->book1D("Pt_Hi", "Pt Hi", 200, 0, 200);

}

//void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
//			  const edm::TriggerResults& triggerResults,
//			  const reco::CaloJet& jet) {



// ***********************************************************
void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			  const reco::CaloJet& jet) {

  LogTrace(metname)<<"[JetAnalyzer] Analyze Calo Jet";

  jetME->Fill(1);

  // Leading jet
  // Histograms are filled once per event
  if (_leadJetFlag == 1) { 
    if (mEtaFirst) mEtaFirst->Fill (jet.eta());
    if (mPhiFirst) mPhiFirst->Fill (jet.phi());
    if (mEFirst)   mEFirst->Fill (jet.energy());
    if (mPtFirst)  mPtFirst->Fill (jet.pt());
    if (mNJets)    mNJets->Fill (_NJets);

    if (_JetLoPass == 1) {
      if (mEta_Lo) mEta_Lo->Fill (jet.eta());
      if (mPhi_Lo) mPhi_Lo->Fill (jet.phi());
      if (mPt_Lo)  mPt_Lo->Fill (jet.pt());
    }
    
    if (_JetHiPass == 1) {
      if (mEta_Hi) mEta_Hi->Fill (jet.eta());
      if (mPhi_Hi) mPhi_Hi->Fill (jet.phi());
      if (mPt_Hi)  mPt_Hi->Fill (jet.pt());
    }
  }


  if (mEta)  mEta->Fill (jet.eta());
  if (mPhi)  mPhi->Fill (jet.phi());
  if (mE)    mE->Fill (jet.energy());
  if (mP)    mP->Fill (jet.p());
  if (mPt)   mPt->Fill (jet.pt());
  if (mPt_1) mPt_1->Fill (jet.pt());
  if (mPt_2) mPt_2->Fill (jet.pt());
  if (mPt_3) mPt_3->Fill (jet.pt());
  if (mMass) mMass->Fill (jet.mass());
  if (mConstituents) mConstituents->Fill (jet.nConstituents());

  if (mMaxEInEmTowers)  mMaxEInEmTowers->Fill (jet.maxEInEmTowers());
  if (mMaxEInHadTowers) mMaxEInHadTowers->Fill (jet.maxEInHadTowers());
  if (mHadEnergyInHO)   mHadEnergyInHO->Fill (jet.hadEnergyInHO());
  if (mHadEnergyInHB)   mHadEnergyInHB->Fill (jet.hadEnergyInHB());
  if (mHadEnergyInHF)   mHadEnergyInHF->Fill (jet.hadEnergyInHF());
  if (mHadEnergyInHE)   mHadEnergyInHE->Fill (jet.hadEnergyInHE());
  if (mEmEnergyInEB)    mEmEnergyInEB->Fill (jet.emEnergyInEB());
  if (mEmEnergyInEE)    mEmEnergyInEE->Fill (jet.emEnergyInEE());
  if (mEmEnergyInHF)    mEmEnergyInHF->Fill (jet.emEnergyInHF());
  if (mEnergyFractionHadronic) mEnergyFractionHadronic->Fill (jet.energyFractionHadronic());
  if (mEnergyFractionEm)       mEnergyFractionEm->Fill (jet.emEnergyFraction());
  if (mN90)  mN90->Fill (jet.n90());  

  if (fabs(jet.eta()) <= 1.3) {
    if (mPt_Barrel)   mPt_Barrel->Fill (jet.pt());
    if (mPhi_Barrel)  mPhi_Barrel->Fill (jet.phi());
    if (mE_Barrel)    mE_Barrel->Fill (jet.energy());
  }
  if ( (fabs(jet.eta()) > 1.3) && (fabs(jet.eta()) <= 3) ) {
    if (mPt_EndCap)   mPt_EndCap->Fill (jet.pt());
    if (mPhi_EndCap)  mPhi_EndCap->Fill (jet.phi());
    if (mE_EndCap)    mE_EndCap->Fill (jet.energy());
  }
  if (fabs(jet.eta()) > 3.0) {
    if (mPt_Forward)   mPt_Forward->Fill (jet.pt());
    if (mPhi_Forward)  mPhi_Forward->Fill (jet.phi());
    if (mE_Forward)    mE_Forward->Fill (jet.energy());
  }


}
