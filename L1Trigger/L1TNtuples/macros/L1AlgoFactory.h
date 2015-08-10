#ifndef L1AlgoFactory_h
#define L1AlgoFactory_h

#include "L1AnalysisGMTDataFormat.h"
#include "L1AnalysisGCTDataFormat.h"
#include "L1AnalysisGTDataFormat.h"

class L1AlgoFactory{
 public:
  L1AlgoFactory();
  L1AlgoFactory(L1Analysis::L1AnalysisGTDataFormat *gt, L1Analysis::L1AnalysisGMTDataFormat *gmt, L1Analysis::L1AnalysisGCTDataFormat *gct);

  void setL1JetCorrection(Bool_t isL1JetCorr) {theL1JetCorrection = isL1JetCorr;}
  void setHF(Bool_t isHF) {noHF = isHF;}
  void setTau(Bool_t isTauInJet) {NOTauInJets = isTauInJet;}

  void SingleMuPt(Float_t& ptcut, Int_t qualmin=4);
  void SingleMuEta2p1Pt(Float_t& ptcut);
  void DoubleMuPt(Float_t& mu1pt, Float_t& mu2pt, Bool_t isHighQual = false, Bool_t isER = false);
  void DoubleMuXOpenPt(Float_t& cut);
  void OniaPt(Float_t& ptcut1, Float_t& ptcut2, Int_t delta);
  void Onia2015Pt(Float_t& ptcut1, Float_t& ptcut2, Bool_t isER, Bool_t isOS, Int_t delta);
  void TripleMuPt(Float_t& mu1pt, Float_t& mu2pt, Float_t& mu3pt, Int_t qualmin = 4);
  void QuadMuPt(Float_t& mu1pt, Float_t& mu2pt, Float_t& mu3pt, Float_t& mu4pt, Int_t qualmin = 4);

  void SingleEGPt(Float_t& ptcut, Bool_t isIsolated = false);
  void SingleEGEta2p1Pt(Float_t& ptcut, Bool_t isIsolated = false);
  void DoubleEGPt(Float_t& ele1pt, Float_t& ele2pt, Bool_t isIsolated = false, Bool_t isER = false);
  void TripleEGPt(Float_t& ele1pt, Float_t& ele2pt, Float_t& ele3pt);

  void SingleJetPt(Float_t& ptcut, Bool_t isCentral = false);
  void DoubleJetPt(Float_t& cut1, Float_t& cut2, Bool_t isCentral = false);
  void DoubleJet_Eta1p7_deltaEta4Pt(Float_t& cut1, Float_t& cut2 );
  void DoubleTauJetEta2p17Pt(Float_t& cut1, Float_t& cut2, Bool_t isIsolated = false);
  void TripleJetPt(Float_t& cut1, Float_t& cut2, Float_t& cut3, Bool_t isCentral = false);
  Bool_t TripleJet_VBF(Float_t jet1, Float_t jet2, Float_t jet3 );
  void QuadJetPt(Float_t& cut1, Float_t& cut2, Float_t& cut3, Float_t& cut4, Bool_t isCentral = false);

  void Mu_EGPt(Float_t& mucut, Float_t& EGcut, Bool_t isIsolated = false, Int_t qualmin=4);
  void DoubleMu_EGPt(Float_t& mucut, Float_t& EGcut, Bool_t isMuHighQual = false );
  void Mu_DoubleEGPt(Float_t& mucut, Float_t& EGcut );

  void Muer_JetCentralPt(Float_t& mucut, Float_t& jetcut);
  void Mu_JetCentral_deltaPt(Float_t& mucut, Float_t& jetcut);
  void Mu_DoubleJetCentralPt(Float_t& mucut, Float_t& jetcut);
  void Mu_HTTPt(Float_t& mucut, Float_t& HTcut );
  void Muer_ETMPt(Float_t& mucut, Float_t& ETMcut );
  void Muer_TauJetEta2p17Pt(Float_t& mucut, Float_t& taucut);
  void Muer_ETM_HTTPt(Float_t& mucut, Float_t& ETMcut, Float_t& HTTcut);
  void Muer_ETM_JetCPt(Float_t& mucut, Float_t& ETMcut, Float_t& jetcut);

  void SingleEG_Eta2p1_HTTPt(Float_t& egcut, Float_t& HTTcut, Bool_t isIsolated = false);
  void EG_FwdJetPt(Float_t& EGcut, Float_t& FWcut);
  void EG_DoubleJetCentralPt(Float_t& EGcut, Float_t& jetcut);
  void EGer_TripleJetCentralPt(Float_t& EGcut, Float_t& jetcut);
  void DoubleEG_HTPt(Float_t& EGcut, Float_t& HTcut);
  void IsoEGer_TauJetEta2p17Pt(Float_t& egcut, Float_t& taucut);
  void Jet_MuOpen_Mu_dPhiMuMu1Pt(Float_t& jetcut, Float_t& mucut);
  void Jet_MuOpen_EG_dPhiMuEG1Pt(Float_t& jetcut, Float_t& egcut);

  void DoubleJetCentral_ETMPt(Float_t& jetcut1, Float_t& jetcut2, Float_t& ETMcut);
  void QuadJetCentral_TauJetPt(Float_t& jetcut, Float_t& taucut);
  void DoubleJetC_deltaPhi7_HTTPt(Float_t& jetcut, Float_t& httcut);

  void ETMVal(Float_t& ETMcut);
  void HTTVal(Float_t& HTTcut);
  void HTMVal(Float_t& HTMcut);
  void ETTVal(Float_t& ETTcut);
  void ETMVal_NoQCD(Float_t& ETMcut);

  Bool_t SingleMu(Float_t ptcut, Int_t qualmin=4);
  Bool_t SingleMuEta2p1(Float_t ptcut);
  Bool_t DoubleMu(Float_t mu1pt, Float_t mu2pt, Bool_t isHighQual = false, Bool_t isER = false);
  Bool_t DoubleMuXOpen(Float_t mu1pt);
  Bool_t Onia(Float_t mu1pt, Float_t mu2pt, Int_t delta);
  Bool_t Onia2015(Float_t mu1pt, Float_t mu2pt, Bool_t isER, Bool_t isOS, Int_t delta);
  Bool_t TripleMu(Float_t mu1pt, Float_t mu2pt, Float_t mu3pt, Int_t qualmin);
  Bool_t QuadMu(Float_t mu1pt, Float_t mu2pt, Float_t mu3pt, Float_t mu4pt, Int_t qualmin);

  Bool_t SingleEG(Float_t ptcut, Bool_t isIsolated = false);
  Bool_t SingleEGEta2p1(Float_t ptcut, Bool_t isIsolated = false);
  Bool_t DoubleEG(Float_t ptcut1, Float_t ptcut2, Bool_t isIsolated = false);
  Bool_t TripleEG(Float_t ptcut1, Float_t ptcut2, Float_t ptcut3);

  Bool_t SingleJet(Float_t ptcut, Bool_t isCentral = false);
  Bool_t DoubleJet(Float_t cut1, Float_t cut2, Bool_t isCentral = false);
  Bool_t DoubleJet_Eta1p7_deltaEta4(Float_t cut1, Float_t cut2 );
  Bool_t DoubleTauJetEta2p17(Float_t cut1, Float_t cut2, Bool_t isIsolated = false);
  Bool_t TripleJet(Float_t cut1, Float_t cut2, Float_t cut3, Bool_t isCentral = false);
  Bool_t QuadJet(Float_t cut1, Float_t cut2, Float_t cut3, Float_t cut4, Bool_t isCentral);

  Bool_t ETM(Float_t ETMcut);
  Bool_t HTT(Float_t HTTcut);
  Bool_t HTM(Float_t HTMcut);
  Bool_t ETT(Float_t ETTcut);
  Bool_t ETM_NoQCD(Float_t ETMcut);

  Bool_t Mu_EG(Float_t mucut, Float_t EGcut, Bool_t isIsolated = false, Int_t qualmin=4);
  Bool_t DoubleMu_EG(Float_t mucut, Float_t EGcut, Bool_t isMuHighQual = false);
  Bool_t Mu_DoubleEG(Float_t mucut, Float_t EGcut);

  Bool_t Muer_JetCentral(Float_t mucut, Float_t jetcut);
  Bool_t Mu_JetCentral_delta(Float_t mucut, Float_t jetcut);
  Bool_t Mu_DoubleJetCentral(Float_t mucut, Float_t jetcut);
  Bool_t Mu_HTT(Float_t mucut, Float_t HTcut);
  Bool_t Muer_ETM(Float_t mucut, Float_t ETMcut);
  Bool_t Muer_TauJetEta2p17(Float_t mucut, Float_t taucut);
  Bool_t Muer_ETM_HTT(Float_t mucut, Float_t ETMcut, Float_t HTTcut);
  Bool_t Muer_ETM_JetC(Float_t mucut, Float_t ETMcut, Float_t jetcut);

  Bool_t SingleEG_Eta2p1_HTT(Float_t egcut, Float_t HTTcut, Bool_t isIsolated = false);
  Bool_t EG_FwdJet(Float_t EGcut, Float_t FWcut);
  Bool_t EG_DoubleJetCentral(Float_t EGcut, Float_t jetcut);
  Bool_t EGer_TripleJetCentral(Float_t EGcut, Float_t jetcut);
  Bool_t DoubleEG_HT(Float_t EGcut, Float_t HTcut);
  Bool_t IsoEGer_TauJetEta2p17(Float_t egcut, Float_t taucut);
  Bool_t Jet_MuOpen_Mu_dPhiMuMu1(Float_t jetcut, Float_t mucut);
  Bool_t Jet_MuOpen_EG_dPhiMuEG1(Float_t jetcut, Float_t egcut);

  Bool_t DoubleJetCentral_ETM(Float_t jetcut1, Float_t jetcut2, Float_t ETMcut);
  Bool_t QuadJetCentral_TauJet(Float_t jetcut, Float_t taucut);
  Bool_t DoubleJetC_deltaPhi7_HTT(Float_t jetcut, Float_t httcut);

  inline Bool_t correlateInPhi(Int_t jetphi, Int_t muphi, Int_t delta=1);
  inline Bool_t correlateInEta(Int_t mueta, Int_t jeteta, Int_t delta=1);
  Double_t CorrectedL1JetPtByGCTregions(Double_t JetiEta,Double_t JetPt);
  Int_t etaMuIdx(Double_t eta);
  Int_t phiINjetCoord(Double_t phi);
  Int_t etaINjetCoord(Double_t eta);
  inline Double_t degree(Double_t radian);

 private:
  Bool_t theL1JetCorrection;

  Bool_t noHF;
  Bool_t NOTauInJets;

  L1Analysis::L1AnalysisGCTDataFormat *gct_;
  L1Analysis::L1AnalysisGMTDataFormat *gmt_;
  L1Analysis::L1AnalysisGTDataFormat  *gt_;

};

const size_t PHIBINS = 18;
const Double_t PHIBIN[] = {10,30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350};
const size_t ETABINS = 23;
const Double_t ETABIN[] = {-5.,-4.5,-4.,-3.5,-3.,-2.172,-1.74,-1.392,-1.044,-0.696,-0.348,0.,0.348,0.696,1.044,1.392,1.74,2.172,3.,3.5,4.,4.5,5.};

const size_t ETAMUBINS = 65;
const Double_t ETAMU[] = { -2.45,-2.4,-2.35,-2.3,-2.25,-2.2,-2.15,-2.1,-2.05,-2,-1.95,-1.9,-1.85,-1.8,-1.75,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45 };

L1AlgoFactory::L1AlgoFactory()
{
}

L1AlgoFactory::L1AlgoFactory(L1Analysis::L1AnalysisGTDataFormat *gt, L1Analysis::L1AnalysisGMTDataFormat *gmt, L1Analysis::L1AnalysisGCTDataFormat *gct)
{
  gt_ = gt;
  gmt_ = gmt;
  gct_ = gct;
  noHF = false;
  NOTauInJets = false;
}

Bool_t L1AlgoFactory::SingleMu(Float_t ptcut, Int_t qualmin) {
  Float_t tmp_cut = -10.;
  SingleMuPt(tmp_cut,qualmin);
  if(tmp_cut >= ptcut) return true;
  return false;
}

Bool_t L1AlgoFactory::SingleMuEta2p1(Float_t ptcut) {
  Float_t tmp_cut = -10.;
  SingleMuEta2p1Pt(tmp_cut);
  if(tmp_cut >= ptcut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleMu(Float_t mu1pt, Float_t mu2pt, Bool_t isHighQual, Bool_t isER) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  DoubleMuPt(tmp_cut1,tmp_cut2,isHighQual,isER);
  if(tmp_cut1 >= mu1pt && tmp_cut2 >= mu2pt) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleMuXOpen(Float_t mu1pt) {
  Float_t tmp_cut = -10.;
  DoubleMuXOpenPt(tmp_cut);
  if(tmp_cut >= mu1pt) return true;
  return false;
}

Bool_t L1AlgoFactory::Onia(Float_t mu1pt, Float_t mu2pt, Int_t delta) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  OniaPt(tmp_cut1,tmp_cut2,delta);
  if(tmp_cut1 >= mu1pt && tmp_cut2 >= mu2pt) return true;
  return false;
}

Bool_t L1AlgoFactory::Onia2015(Float_t mu1pt, Float_t mu2pt, Bool_t isER, Bool_t isOS, Int_t delta) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  Onia2015Pt(tmp_cut1,tmp_cut2,isER,isOS,delta);
  if(tmp_cut1 >= mu1pt && tmp_cut2 >= mu2pt) return true;
  return false;
}

Bool_t L1AlgoFactory::TripleMu(Float_t mu1pt, Float_t mu2pt, Float_t mu3pt, Int_t qualmin) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  Float_t tmp_cut3 = -10.;
  TripleMuPt(tmp_cut1,tmp_cut2,tmp_cut3,qualmin);
  if(tmp_cut1 >= mu1pt && tmp_cut2 >= mu2pt &&  tmp_cut3 >= mu3pt) return true;
  return false;
}

Bool_t L1AlgoFactory::QuadMu(Float_t mu1pt, Float_t mu2pt, Float_t mu3pt, Float_t mu4pt, Int_t qualmin) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  Float_t tmp_cut3 = -10.;
  Float_t tmp_cut4 = -10.;
  QuadMuPt(tmp_cut1,tmp_cut2,tmp_cut3,tmp_cut4,qualmin);
  if(tmp_cut1 >= mu1pt && tmp_cut2 >= mu2pt &&  tmp_cut3 >= mu3pt && tmp_cut4 >= mu4pt) return true;
  return false;
}

Bool_t L1AlgoFactory::SingleEG(Float_t ptcut, Bool_t isIsolated) {
  Float_t tmp_cut1 = -10.;
  SingleEGPt(tmp_cut1,isIsolated);
  if(tmp_cut1 >= ptcut) return true;
  return false;
}

Bool_t L1AlgoFactory::SingleEGEta2p1(Float_t ptcut, Bool_t isIsolated) {
  Float_t tmp_cut1 = -10.;
  SingleEGEta2p1Pt(tmp_cut1,isIsolated);
  if(tmp_cut1 >= ptcut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleEG(Float_t ptcut1, Float_t ptcut2, Bool_t isIsolated) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  DoubleEGPt(tmp_cut1,tmp_cut2,isIsolated);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2) return true;
  return false;
}

Bool_t L1AlgoFactory::TripleEG(Float_t ptcut1, Float_t ptcut2, Float_t ptcut3) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  Float_t tmp_cut3 = -10.;
  TripleEGPt(tmp_cut1,tmp_cut2,tmp_cut3);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2 && tmp_cut3 >= ptcut3) return true;
  return false;
}

Bool_t L1AlgoFactory::SingleJet(Float_t ptcut, Bool_t isCentral) {
  Float_t tmp_cut1 = -10.;
  SingleJetPt(tmp_cut1,isCentral);
  if(tmp_cut1 >= ptcut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleJet(Float_t ptcut1, Float_t ptcut2, Bool_t isCentral) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  DoubleJetPt(tmp_cut1,tmp_cut2,isCentral);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleJet_Eta1p7_deltaEta4(Float_t ptcut1, Float_t ptcut2) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  DoubleJet_Eta1p7_deltaEta4Pt(tmp_cut1,tmp_cut2);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleTauJetEta2p17(Float_t ptcut1, Float_t ptcut2, Bool_t isIsolated) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  DoubleTauJetEta2p17Pt(tmp_cut1,tmp_cut2,isIsolated);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2) return true;
  return false;
}

Bool_t L1AlgoFactory::TripleJet(Float_t ptcut1, Float_t ptcut2, Float_t ptcut3, Bool_t isCentral) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  Float_t tmp_cut3 = -10.;
  TripleJetPt(tmp_cut1,tmp_cut2,tmp_cut3,isCentral);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2 && tmp_cut3 >= ptcut3) return true;
  return false;
}

Bool_t L1AlgoFactory::QuadJet(Float_t ptcut1, Float_t ptcut2, Float_t ptcut3, Float_t ptcut4, Bool_t isCentral) {
  Float_t tmp_cut1 = -10.;
  Float_t tmp_cut2 = -10.;
  Float_t tmp_cut3 = -10.;
  Float_t tmp_cut4 = -10.;
  QuadJetPt(tmp_cut1,tmp_cut2,tmp_cut3,tmp_cut4,isCentral);
  if(tmp_cut1 >= ptcut1 && tmp_cut2 >= ptcut2 && tmp_cut3 >= ptcut3 && tmp_cut4 >= ptcut4) return true;
  return false;
}

Bool_t L1AlgoFactory::ETM(Float_t ETMcut) {
  Float_t tmp_cut = -10.;
  ETMVal(tmp_cut);
  if(tmp_cut >= ETMcut) return true;
  return false;
}

Bool_t L1AlgoFactory::HTT(Float_t HTTcut) {
  Float_t tmp_cut = -10.;
  HTTVal(tmp_cut);
  if(tmp_cut >= HTTcut) return true;
  return false;
}

Bool_t L1AlgoFactory::HTM(Float_t HTMcut) {
  Float_t tmp_cut = -10.;
  HTMVal(tmp_cut);
  if(tmp_cut >= HTMcut) return true;
  return false;
}

Bool_t L1AlgoFactory::ETT(Float_t ETTcut) {
  Float_t tmp_cut = -10.;
  ETTVal(tmp_cut);
  if(tmp_cut >= ETTcut) return true;
  return false;
}

Bool_t L1AlgoFactory::ETM_NoQCD(Float_t ETMcut) {
  Float_t tmp_cut = -10.;
  ETMVal_NoQCD(tmp_cut);
  if(tmp_cut >= ETMcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Mu_EG(Float_t mucut, Float_t EGcut, Bool_t isIsolated, Int_t qualmin) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_elecut = -10.;
  Mu_EGPt(tmp_mucut,tmp_elecut,isIsolated,qualmin);
  if(tmp_mucut >= mucut && tmp_elecut >= EGcut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleMu_EG(Float_t mucut, Float_t EGcut, Bool_t isMuHighQual) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_elecut = -10.;
  DoubleMu_EGPt(tmp_mucut,tmp_elecut,isMuHighQual);
  if(tmp_mucut >= mucut && tmp_elecut >= EGcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Mu_DoubleEG(Float_t mucut, Float_t EGcut) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_elecut = -10.;
  Mu_DoubleEGPt(tmp_mucut,tmp_elecut);
  if(tmp_mucut >= mucut && tmp_elecut >= EGcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Muer_JetCentral(Float_t mucut, Float_t jetcut) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_jetcut = -10.;
  Muer_JetCentralPt(tmp_mucut,tmp_jetcut);
  if(tmp_mucut >= mucut && tmp_jetcut >= jetcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Mu_JetCentral_delta(Float_t mucut, Float_t jetcut) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_jetcut = -10.;
  Mu_JetCentral_deltaPt(tmp_mucut,tmp_jetcut);
  if(tmp_mucut >= mucut && tmp_jetcut >= jetcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Mu_DoubleJetCentral(Float_t mucut, Float_t jetcut) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_jetcut = -10.;
  Mu_DoubleJetCentralPt(tmp_mucut,tmp_jetcut);
  if(tmp_mucut >= mucut && tmp_jetcut >= jetcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Mu_HTT(Float_t mucut, Float_t HTcut) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_HTcut = -10.;
  Mu_HTTPt(tmp_mucut,tmp_HTcut);
  if(tmp_mucut >= mucut && tmp_HTcut >= HTcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Muer_ETM(Float_t mucut, Float_t ETMcut) {
  Float_t tmp_mucut = -10.;
  Float_t tmp_ETMcut = -10.;
  Muer_ETMPt(tmp_mucut,tmp_ETMcut);
  if(tmp_mucut >= mucut && tmp_ETMcut >= ETMcut) return true;
  return false;
}

Bool_t L1AlgoFactory::SingleEG_Eta2p1_HTT(Float_t egcut, Float_t HTTcut, Bool_t isIsolated) {
  Float_t tmp_egcut = -10.;
  Float_t tmp_HTTcut = -10.;
  SingleEG_Eta2p1_HTTPt(tmp_egcut,tmp_HTTcut,isIsolated);
  if(tmp_egcut >= egcut && tmp_HTTcut >= HTTcut) return true;
  return false;
}

Bool_t L1AlgoFactory::EG_FwdJet(Float_t egcut, Float_t FWcut) {
  Float_t tmp_egcut = -10.;
  Float_t tmp_FWcut = -10.;
  EG_FwdJetPt(tmp_egcut,tmp_FWcut);
  if(tmp_egcut >= egcut && tmp_FWcut >= FWcut) return true;
  return false;
}

Bool_t L1AlgoFactory::EG_DoubleJetCentral(Float_t egcut, Float_t jetcut) {
  Float_t tmp_egcut = -10.;
  Float_t tmp_jetcut = -10.;
  EG_DoubleJetCentralPt(tmp_egcut,tmp_jetcut);
  if(tmp_egcut >= egcut && tmp_jetcut >= jetcut) return true;
  return false;
}

Bool_t L1AlgoFactory::EGer_TripleJetCentral(Float_t egcut, Float_t jetcut) {
  Float_t tmp_egcut = -10.;
  Float_t tmp_jetcut = -10.;
  EGer_TripleJetCentralPt(tmp_egcut,tmp_jetcut);
  if(tmp_egcut >= egcut && tmp_jetcut >= jetcut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleEG_HT(Float_t egcut, Float_t HTcut) {
  Float_t tmp_egcut = -10.;
  Float_t tmp_HTcut = -10.;
  DoubleEG_HTPt(tmp_egcut,tmp_HTcut);
  if(tmp_egcut >= egcut && tmp_HTcut >= HTcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Jet_MuOpen_Mu_dPhiMuMu1(Float_t jetcut, Float_t mucut) {
  Float_t tmp_jetcut = -10.;
  Float_t tmp_mucut = -10.;
  Jet_MuOpen_Mu_dPhiMuMu1Pt(tmp_jetcut,tmp_mucut);
  if(tmp_jetcut >= jetcut && tmp_mucut >= mucut) return true;
  return false;
}

Bool_t L1AlgoFactory::Jet_MuOpen_EG_dPhiMuEG1(Float_t jetcut, Float_t egcut) {
  Float_t tmp_jetcut = -10.;
  Float_t tmp_egcut = -10.;
  Jet_MuOpen_EG_dPhiMuEG1Pt(tmp_jetcut,tmp_egcut);
  if(tmp_jetcut >= jetcut && tmp_egcut >= egcut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleJetCentral_ETM(Float_t jetcut1, Float_t jetcut2, Float_t ETMcut) {
  Float_t tmp_jetcut1 = -10.;
  Float_t tmp_jetcut2 = -10.;
  Float_t tmp_ETMcut  = -10.;
  DoubleJetCentral_ETMPt(tmp_jetcut1,tmp_jetcut2,tmp_ETMcut);
  if(tmp_jetcut1 >= jetcut1 && tmp_jetcut2 >= jetcut2 && tmp_ETMcut >= ETMcut) return true;
  return false;
}

Bool_t L1AlgoFactory::Muer_TauJetEta2p17(Float_t mucut, Float_t taucut){
  Float_t tmp_mucut  = -10.;
  Float_t tmp_taucut = -10.;
  Muer_TauJetEta2p17Pt(tmp_mucut, tmp_taucut);
  if(tmp_mucut >= mucut && tmp_taucut >= taucut) return true;
  return false;
}

Bool_t L1AlgoFactory::IsoEGer_TauJetEta2p17(Float_t egcut, Float_t taucut){
  Float_t tmp_egcut  = -10.;
  Float_t tmp_taucut = -10.;
  IsoEGer_TauJetEta2p17Pt(tmp_egcut, tmp_taucut);
  if(tmp_egcut >= egcut && tmp_taucut >= taucut) return true;
  return false;
}

Bool_t L1AlgoFactory::QuadJetCentral_TauJet(Float_t jetcut, Float_t taucut){
  Float_t tmp_jetcut = -10.;
  Float_t tmp_taucut = -10.;
  QuadJetCentral_TauJetPt(tmp_jetcut,tmp_taucut);
  if(tmp_jetcut >= jetcut && tmp_taucut >= taucut) return true;
  return false;
}

Bool_t L1AlgoFactory::DoubleJetC_deltaPhi7_HTT(Float_t jetcut, Float_t HTTcut){
  Float_t tmp_jetcut = -10.;
  Float_t tmp_httcut = -10.;
  DoubleJetC_deltaPhi7_HTTPt(tmp_jetcut,tmp_httcut);
  if(tmp_jetcut >= jetcut && tmp_httcut >= HTTcut) return true;
  return false;
}

Bool_t L1AlgoFactory:: Muer_ETM_HTT(Float_t mucut, Float_t ETMcut, Float_t HTTcut){
  Float_t tmp_mucut  = -10.;
  Float_t tmp_ETMcut = -10.;
  Float_t tmp_HTTcut = -10.;
  Muer_ETM_HTTPt(tmp_mucut,tmp_ETMcut,tmp_HTTcut);
  if(tmp_mucut >= mucut && tmp_ETMcut >= ETMcut && tmp_HTTcut >= HTTcut) return true;
  return false;
}

Bool_t L1AlgoFactory:: Muer_ETM_JetC(Float_t mucut, Float_t ETMcut, Float_t jetcut){
  Float_t tmp_mucut  = -10.;
  Float_t tmp_ETMcut = -10.;
  Float_t tmp_jetcut = -10.;
  Muer_ETM_JetCPt(tmp_mucut,tmp_ETMcut,tmp_jetcut);
  if(tmp_mucut >= mucut && tmp_ETMcut >= ETMcut && tmp_jetcut >= jetcut) return true;
  return false;
}

inline Bool_t L1AlgoFactory::correlateInPhi(Int_t jetphi, Int_t muphi, Int_t delta){
  return fabs(muphi-jetphi) < fabs(1 + delta) || fabs(muphi-jetphi) > fabs(PHIBINS - 1 - delta) ;
}

inline Bool_t L1AlgoFactory::correlateInEta(Int_t mueta, Int_t jeteta, Int_t delta) {
  return fabs(mueta-jeteta) < 1 + delta;
}

Int_t L1AlgoFactory::etaMuIdx(Double_t eta) {
  size_t etaIdx = 0.;
  for (size_t idx=0; idx<ETAMUBINS; idx++) {
    if (eta>=ETAMU[idx] and eta<ETAMU[idx+1])
      etaIdx = idx;
  }

  return int(etaIdx);
}

Int_t L1AlgoFactory::phiINjetCoord(Double_t phi) {
  size_t phiIdx = 0;
  Double_t phidegree = degree(phi);
  for (size_t idx=0; idx<PHIBINS; idx++) {
    if (phidegree>=PHIBIN[idx] and phidegree<PHIBIN[idx+1])
      phiIdx = idx;
    else if (phidegree>=PHIBIN[PHIBINS-1] || phidegree<=PHIBIN[0])
      phiIdx = idx;
  }
  phiIdx = phiIdx + 1;
  if (phiIdx == 18)  phiIdx = 0;

  return int(phiIdx);
}

Int_t L1AlgoFactory::etaINjetCoord(Double_t eta) {
  size_t etaIdx = 0.;
  for (size_t idx=0; idx<ETABINS; idx++) {
    if (eta>=ETABIN[idx] and eta<ETABIN[idx+1])
      etaIdx = idx;
  }

  return int(etaIdx);
}

inline Double_t L1AlgoFactory::degree(Double_t radian) {
  if (radian<0.)
    return 360.+(radian/acos(-1.)*180.);
  else
    return radian/acos(-1.)*180.;
}

// correction for RCT->GCT bins (from HCAL January 2012)
// HF from 29-41, first 3 HF trigger towers 3 iEtas, last highest eta HF trigger tower 4 iEtas; each trigger tower is 0.5 eta, RCT iEta from 0->21 (left->right)
Double_t L1AlgoFactory::CorrectedL1JetPtByGCTregions(Double_t JetiEta,Double_t JetPt) {

  static const Double_t JetRCTHFiEtacorr[]  = {0.965,0.943,0.936,0.929}; // from HF iEta=29 to 41 (smaller->higher HF iEta)
  Double_t JetPtcorr   = JetPt;

  if (theL1JetCorrection) {

    if ((JetiEta>=7 && JetiEta<=14)) {
      JetPtcorr = JetPt * 1.05;
    }

    if ((JetiEta>=4 && JetiEta<=6) || (JetiEta>=15 && JetiEta<=17)) {
      JetPtcorr = JetPt * 0.95;
    }

    if (JetiEta==0 || JetiEta==21) {
      JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[3]));
    }
    else if (JetiEta==1 || JetiEta==20) {
      JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[2]));
    }
    else if (JetiEta==2 || JetiEta==19) {
      JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[1]));
    }
    else if (JetiEta==3 || JetiEta==18) {
      JetPtcorr = JetPt * (1+(1-JetRCTHFiEtacorr[0]));
    }
  }

  return JetPtcorr;
}

void L1AlgoFactory::SingleMuPt(Float_t& ptcut, Int_t qualmin) {

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 1) return;

  Float_t ptmax = -10.;

  for(Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];             //    BX = 0, +/- 1 or +/- 2
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];                       
    Int_t qual = gmt_ -> Qual[imu];             
    if( qual < qualmin) continue;
    if(pt >= ptmax) ptmax = pt;
  }

  ptcut = ptmax;
  return;
}

void L1AlgoFactory::SingleMuEta2p1Pt(Float_t& ptcut) {

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 1) return;

  Float_t ptmax = -10.;

  for(Int_t imu=0; imu < Nmu; imu++) { 
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( qual < 4) continue;
    Float_t eta = gmt_  -> Eta[imu];        
    if(fabs(eta) > 2.1) continue;
    if(pt >= ptmax) ptmax = pt;
  }

  ptcut = ptmax;
  return;
}

void L1AlgoFactory::DoubleMuPt(Float_t& cut1, Float_t& cut2, Bool_t isHighQual, Bool_t isER) {

  Float_t mu1ptmax = -10.;
  Float_t mu2ptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 2) return;

  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if(qual < 4  && qual != 3 ) continue;
    if(qual < 4  && isHighQual ) continue;
    Float_t eta = gmt_ -> Eta[imu];            
    if(isER && fabs(eta) > 2.1) continue;

    if(pt >= mu1ptmax)
      {
	mu2ptmax = mu1ptmax;
	mu1ptmax = pt;
      }
    else if(pt >= mu2ptmax) mu2ptmax = pt;
  }

  if(mu2ptmax >= 0.){
    cut1 = mu1ptmax;
    cut2 = mu2ptmax;
  }

  return;
}

void L1AlgoFactory::DoubleMuXOpenPt(Float_t& cut) {

  Int_t n2=0;
  Float_t ptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  for(Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( (qual >= 5 || qual == 3 ) && pt >= ptmax ){
      ptmax = pt;
    }
    if(pt >= 0.) n2++;
  }

  if(n2>=2) cut = ptmax;
  else cut = -10.;

  return;
}

void L1AlgoFactory::OniaPt(Float_t& ptcut1, Float_t& ptcut2, Int_t delta) {

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 2) return;

  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;
  Float_t corr = false;

  std::vector<std::pair<Float_t,Float_t> > muonPairs;

  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < 4) continue;
    Float_t eta = gmt_  -> Eta[imu];        
    if (fabs(eta) > 2.1) continue;

    Int_t ieta1 = etaMuIdx(eta);

    for (Int_t imu2=0; imu2 < Nmu; imu2++) {
      if (imu2 == imu) continue;
      Int_t bx2 = gmt_ -> CandBx[imu2];		
      if (bx2 != 0) continue;
      Float_t pt2 = gmt_ -> Pt[imu2];			
      Int_t qual2 = gmt_ -> Qual[imu2];        
      if ( qual2 < 4) continue;
      Float_t eta2 = gmt_  -> Eta[imu2];        
      if (fabs(eta2) > 2.1) continue;
      Int_t ieta2 = etaMuIdx(eta2);

      Float_t deta = ieta1 - ieta2; 
      if ( fabs(deta) <= delta){
	corr = true;
	muonPairs.push_back(std::pair<Float_t,Float_t>(pt,pt2));
      }

    }
  }

  if(corr){
    std::vector<std::pair<Float_t,Float_t> >::const_iterator muonPairIt  = muonPairs.begin();
    std::vector<std::pair<Float_t,Float_t> >::const_iterator muonPairEnd = muonPairs.end();
    for (; muonPairIt != muonPairEnd; ++muonPairIt) {
      Float_t pt1 = muonPairIt->first;
      Float_t pt2 = muonPairIt->second;
      
      if ( pt1 > maxpt1 || (fabs(maxpt1-pt1)<10E-2 && pt2>maxpt2) ) 
	{
	  maxpt1 = pt1;
	  maxpt2 = pt2;
	}
    }

  }

  if(corr && maxpt2 >= 0.){
    ptcut1 = maxpt1;
    ptcut2 = maxpt2;
  }

  return;
}

void L1AlgoFactory::Onia2015Pt(Float_t& ptcut1, Float_t& ptcut2, Bool_t isER, Bool_t isOS, Int_t delta) {

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 2) return;

  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;
  Float_t corr = false;

  std::vector<std::pair<Float_t,Float_t> > muonPairs;

  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < 4) continue;
    Float_t eta = gmt_  -> Eta[imu];        
    if(isER && fabs(eta) > 1.6) continue;
    Int_t ieta1 = etaMuIdx(eta);
    Int_t charge1 = gmt_->Cha[imu];

    for (Int_t imu2=0; imu2 < Nmu; imu2++) {
      if (imu2 == imu) continue;
      Int_t bx2 = gmt_ -> CandBx[imu2];		
      if (bx2 != 0) continue;
      Float_t pt2 = gmt_ -> Pt[imu2];			
      Int_t qual2 = gmt_ -> Qual[imu2];        
      if ( qual2 < 4) continue;
      Float_t eta2 = gmt_  -> Eta[imu2];        
      if(isER && fabs(eta2) > 1.6) continue;
      Int_t ieta2 = etaMuIdx(eta2);
      Int_t charge2 = gmt_->Cha[imu2];

      if(isOS && charge1*charge2 > 0) continue;

      Float_t deta = ieta1 - ieta2; 
      if(fabs(deta) <= delta){
	corr = true;
	muonPairs.push_back(std::pair<Float_t,Float_t>(pt,pt2));
      }

    }
  }

  if(corr){
    std::vector<std::pair<Float_t,Float_t> >::const_iterator muonPairIt  = muonPairs.begin();
    std::vector<std::pair<Float_t,Float_t> >::const_iterator muonPairEnd = muonPairs.end();
    for(; muonPairIt != muonPairEnd; ++muonPairIt){
      Float_t pt1 = muonPairIt->first;
      Float_t pt2 = muonPairIt->second;
      
      if(pt1 > maxpt1 || (fabs(maxpt1-pt1)<10E-2 && pt2>maxpt2) ) 
	{
	  maxpt1 = pt1;
	  maxpt2 = pt2;
	}
    }

  }

  if(corr && maxpt2 >= 0.){
    ptcut1 = maxpt1;
    ptcut2 = maxpt2;
  }

  return;
}

void L1AlgoFactory::TripleMuPt(Float_t& cut1, Float_t& cut2, Float_t& cut3, Int_t qualmin) {

  Float_t mu1ptmax = -10.;
  Float_t mu2ptmax = -10.;
  Float_t mu3ptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 3) return;

  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < qualmin) continue;

    if(pt >= mu1ptmax)
      {
	mu3ptmax = mu2ptmax;
	mu2ptmax = mu1ptmax;
	mu1ptmax = pt;
      }
    else if(pt >= mu2ptmax){
      mu3ptmax = mu2ptmax;
      mu2ptmax = pt;
    }
    else if(pt >= mu3ptmax) mu3ptmax = pt;
  }

  if(mu3ptmax >= 0.){
    cut1 = mu1ptmax;
    cut2 = mu2ptmax;
    cut3 = mu3ptmax;
  }

  return;
}

void L1AlgoFactory::QuadMuPt(Float_t& cut1, Float_t& cut2, Float_t& cut3, Float_t& cut4, Int_t qualmin) {

  Float_t mu1ptmax = -10.;
  Float_t mu2ptmax = -10.;
  Float_t mu3ptmax = -10.;
  Float_t mu4ptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 4) return;

  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < qualmin) continue;

    if(pt >= mu1ptmax)
      {
	mu4ptmax = mu3ptmax;
	mu3ptmax = mu2ptmax;
	mu2ptmax = mu1ptmax;
	mu1ptmax = pt;
      }
    else if(pt >= mu2ptmax){
      mu4ptmax = mu3ptmax;
      mu3ptmax = mu2ptmax;
      mu2ptmax = pt;
    }
    else if(pt >= mu3ptmax){
      mu4ptmax = mu3ptmax;
      mu3ptmax = pt;
    }
    else if(pt >= mu4ptmax) mu4ptmax = pt;
  }

  if(mu4ptmax >= 0.){
    cut1 = mu1ptmax;
    cut2 = mu2ptmax;
    cut3 = mu3ptmax;
    cut4 = mu4ptmax;
  }

  return;
}

void L1AlgoFactory::SingleEGPt(Float_t& cut, Bool_t isIsolated ) {

  Int_t Nele = gt_ -> Nele;
  Float_t ptmax = -10.;

  for(Int_t ue=0; ue < Nele; ue++) {               
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    if(isIsolated && !gt_ -> Isoel[ue]) continue;

    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if(pt >= ptmax) ptmax = pt;
  }

  cut = ptmax;

  return;
}

void L1AlgoFactory::SingleEGEta2p1Pt(Float_t& cut, Bool_t isIsolated ) {

  Int_t Nele = gt_ -> Nele;
  Float_t ptmax = -10.;

  for(Int_t ue=0; ue < Nele; ue++) {               
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    if(isIsolated && !gt_ -> Isoel[ue]) continue;
    Float_t eta = gt_ -> Etael[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16

    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if(pt >= ptmax) ptmax = pt;
  }

  cut = ptmax;

  return;
}

void L1AlgoFactory::DoubleEGPt(Float_t& cut1, Float_t& cut2, Bool_t isIsolated, Bool_t isER ) {

  Int_t Nele = gt_ -> Nele;
  if(Nele < 2) return;

  Float_t ele1ptmax = -10.;
  Float_t ele2ptmax = -10.;

  Float_t ele1Phimax = -1000.;
  Float_t ele1Etamax = -1000.;

  Bool_t EG1_ER = false;
  Bool_t EG2_ER = false;

  Bool_t EG1_isol = false;
  Bool_t EG2_isol = false;

  for(Int_t ue=0; ue < Nele; ue++) {               
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    Float_t phi = gt_ -> Phiel[ue];    // the phi  of the electron
    Float_t eta = gt_ -> Etael[ue];    // the eta  of the electron

    if(fabs(pt-ele1ptmax) < 0.001 && fabs(phi-ele1Phimax) < 0.001 && fabs(eta-ele1Etamax) < 0.001) continue; //to avoid double counting in noniso/relaxiso lists

    if(pt >= ele1ptmax)
      {
	ele2ptmax = ele1ptmax;
	EG2_ER = EG1_ER;
	EG2_isol = EG1_isol;
	ele1ptmax = pt;
	ele1Phimax = phi;
	ele1Etamax = eta;
	if(eta > 4.5 && eta < 16.5) EG1_ER = true;
	else EG1_ER = false;
	EG1_isol = gt_ -> Isoel[ue];
      }
    else if(pt >= ele2ptmax){
      ele2ptmax = pt;
      if(eta > 4.5 && eta < 16.5) EG2_ER = true;
      else EG2_ER = false;
      EG1_isol = gt_ -> Isoel[ue];
    }
  }

  if(isER && (!EG1_ER || !EG2_ER)) return;
  if(isIsolated && (!EG1_isol && !EG2_isol)) return;

  if(ele2ptmax >= 0.){
    cut1 = ele1ptmax;
    cut2 = ele2ptmax;
  }

  return;
}

void L1AlgoFactory::TripleEGPt(Float_t& cut1, Float_t& cut2, Float_t& cut3 ) {

  Int_t Nele = gt_ -> Nele;
  if(Nele < 3) return;

  Float_t ele1ptmax = -10.;
  Float_t ele2ptmax = -10.;
  Float_t ele3ptmax = -10.;

  Float_t ele1Phimax = -1000.;
  Float_t ele1Etamax = -1000.;

  Float_t ele2Phimax = -1000.;
  Float_t ele2Etamax = -1000.;

  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if (bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    Float_t phi = gt_ -> Phiel[ue];    // the rank of the electron
    Float_t eta = gt_ -> Etael[ue];    // the rank of the electron

    if(fabs(pt-ele1ptmax) < 0.001 && fabs(phi-ele1Phimax) < 0.001 && fabs(eta-ele1Etamax) < 0.001) continue; //to avoid double counting in noniso/relaxiso lists
    if(fabs(pt-ele2ptmax) < 0.001 && fabs(phi-ele2Phimax) < 0.001 && fabs(eta-ele2Etamax) < 0.001) continue; //to avoid double counting in noniso/relaxiso lists

    if(pt >= ele1ptmax)
      {
	ele3ptmax = ele2ptmax;
	ele2ptmax = ele1ptmax;
	ele1ptmax = pt;
 	ele1Phimax = phi;
	ele1Etamax = eta;
      }
    else if(pt >= ele2ptmax){
      ele3ptmax = ele2ptmax;
      ele2ptmax = pt;
      ele2Phimax = phi;
      ele2Etamax = eta;
    }
    else if(pt >= ele3ptmax) ele3ptmax = pt;
  }

  if(ele3ptmax >= 0.){
    cut1 = ele1ptmax;
    cut2 = ele2ptmax;
    cut3 = ele3ptmax;
  }

  return;
}

void L1AlgoFactory::SingleJetPt(Float_t& cut, Bool_t isCentral) {

  Float_t ptmax = -10.;
  Int_t Nj = gt_ -> Njet ;
  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isCentral && isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(isCentral && noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if(pt >= ptmax) ptmax = pt;
  }

  cut = ptmax;
  return;
}

void L1AlgoFactory::DoubleJetPt(Float_t& cut1, Float_t& cut2, Bool_t isCentral ) {

  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 2) return;

  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isCentral && isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(isCentral && noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= maxpt1)
      {
	maxpt2 = maxpt1;
	maxpt1 = pt;
      }
    else if(pt >= maxpt2) maxpt2 = pt;
  }

  if(maxpt2 >= 0.){
    cut1 = maxpt1;
    cut2 = maxpt2;
  }

  return;
}

void L1AlgoFactory::DoubleJet_Eta1p7_deltaEta4Pt(Float_t& cut1, Float_t& cut2 ) {

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 2) return;

  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;
  Bool_t corr = false;
  std::vector<std::pair<Float_t,Float_t> > jetPairs;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if (isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    Float_t eta1 = gt_ -> Etajet[ue];
    if (eta1 < 5.5 || eta1 > 16.5) continue;  // eta = 6 - 16

    for(Int_t ve=0; ve < Nj; ve++) {
      if(ve == ue) continue;
      Int_t bx2 = gt_ -> Bxjet[ve];        		
      if(bx2 != 0) continue;
      Bool_t isFwdJet2 = gt_ -> Fwdjet[ve];
      if(isFwdJet2) continue;
      if(NOTauInJets && gt_->Taujet[ve]) continue;
      if(noHF && (gt_->Etajet[ve] < 5 || gt_->Etajet[ve] > 17)) continue;

      Float_t rank2 = gt_ -> Rankjet[ve];
      Float_t pt2 = rank2 * 4;
      Float_t eta2 = gt_ -> Etajet[ve];
      if(eta2 < 5.5 || eta2 > 16.5) continue;  // eta = 6 - 16

      if(correlateInEta((int)eta1, (int)eta2, 4)){
	corr = true;
	jetPairs.push_back(std::pair<Float_t,Float_t>(pt,pt2));
      }

    }
  }

  if(corr){
    std::vector<std::pair<Float_t,Float_t> >::const_iterator jetPairIt  = jetPairs.begin();
    std::vector<std::pair<Float_t,Float_t> >::const_iterator jetPairEnd = jetPairs.end();
    for (; jetPairIt != jetPairEnd; ++jetPairIt) {
      Float_t pt1 = jetPairIt->first;
      Float_t pt2 = jetPairIt->second;
      
      if ( pt1 > maxpt1 || (fabs(maxpt1-pt1)<10E-2 && pt2>maxpt2) ) 
	{
	  maxpt1 = pt1;
	  maxpt2 = pt2;
	}
    }
  }

  cut1 = maxpt1;
  cut2 = maxpt2;

  return;
}

void L1AlgoFactory::DoubleTauJetEta2p17Pt(Float_t& cut1, Float_t& cut2, Bool_t isIsolated) {

  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 2) return;

  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue; 
    Bool_t isTauJet = gt_ -> Taujet[ue];
    if(!isTauJet) continue;
    Float_t rank = gt_ -> Rankjet[ue];    // the rank of the electron
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    Float_t eta = gt_ -> Etajet[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16

    if(pt >= maxpt1)
      {
	maxpt2 = maxpt1;
	maxpt1 = pt;
      }
    else if(pt >= maxpt2) maxpt2 = pt;
  }

  if(maxpt2 >= 0.){
    cut1 = maxpt1;
    cut2 = maxpt2;
  }

  return;
}

void L1AlgoFactory::TripleJetPt(Float_t& cut1, Float_t& cut2, Float_t& cut3, Bool_t isCentral) {

  Float_t jet1ptmax = -10.;
  Float_t jet2ptmax = -10.;
  Float_t jet3ptmax = -10.;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 3) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isFwdJet && isCentral) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;
    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= jet1ptmax)
      {
	jet3ptmax = jet2ptmax;
	jet2ptmax = jet1ptmax;
	jet1ptmax = pt;
      }
    else if(pt >= jet2ptmax){
      jet3ptmax = jet2ptmax;
      jet2ptmax = pt;
    }
    else if(pt >= jet3ptmax) jet3ptmax = pt;
  }

  if(jet3ptmax >= 0.){
    cut1 = jet1ptmax;
    cut2 = jet2ptmax;
    cut3 = jet3ptmax;
  }

  return;
}

//For now, only usable in Menu mode
Bool_t L1AlgoFactory::TripleJet_VBF(Float_t jet1, Float_t jet2, Float_t jet3 ) {

  Bool_t jet=false;        
  Bool_t jetf1=false;           
  Bool_t jetf2=false;   

  Int_t n1=0;
  Int_t n2=0;
  Int_t n3=0;

  Int_t f1=0;
  Int_t f2=0;
  Int_t f3=0;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 3) return false;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;

    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if (isFwdJet) {
      if(pt >= jet1) f1++;
      if(pt >= jet2) f2++;
      if(pt >= jet3) f3++;              
    } 
    else {
      if(pt >= jet1) n1++;
      if(pt >= jet2) n2++;
      if(pt >= jet3) n3++;
    }    
  }

  jet   = ( n1 >= 1 && n2 >= 2 && n3 >= 3 ) ;        
  jetf1 = ( f1 >= 1 && n2 >= 1 && n3 >= 2 ) ;  // numbers change ofcourse    
  jetf2 = ( n1 >= 1 && f2 >= 1 && n3 >= 2 ) ;  

  return ( jet || jetf1 || jetf2 );
}

void L1AlgoFactory::QuadJetPt(Float_t& cut1, Float_t& cut2, Float_t& cut3, Float_t& cut4, Bool_t isCentral){

  Float_t jet1ptmax = -10.;
  Float_t jet2ptmax = -10.;
  Float_t jet3ptmax = -10.;
  Float_t jet4ptmax = -10.;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 4) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isCentral && isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(isCentral && noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= jet1ptmax)
      {
	jet4ptmax = jet3ptmax;
	jet3ptmax = jet2ptmax;
	jet2ptmax = jet1ptmax;
	jet1ptmax = pt;
      }
    else if(pt >= jet2ptmax){
      jet4ptmax = jet3ptmax;
      jet3ptmax = jet2ptmax;
      jet2ptmax = pt;
    }
    else if(pt >= jet3ptmax){
      jet4ptmax = jet3ptmax;
      jet3ptmax = pt;
    }
    else if(pt >= jet4ptmax) jet4ptmax = pt;
  }

  if(jet4ptmax >= 0.){
    cut1 = jet1ptmax;
    cut2 = jet2ptmax;
    cut3 = jet3ptmax;
    cut4 = jet4ptmax;
  }

  return;
}

void L1AlgoFactory::ETMVal(Float_t& ETMcut ) {

  Float_t adc = gt_->RankETM;
  Float_t TheETM = adc/2.;
  ETMcut = TheETM;
  return;
}

void L1AlgoFactory::HTTVal(Float_t& HTTcut) {

  Float_t adc = gt_->RankHTT;
  Float_t TheHTT = adc/2.;
  HTTcut = TheHTT;
  return;
}

void L1AlgoFactory::HTMVal(Float_t& HTMcut) {

  Float_t adc = gt_->RankHTM ;
  Float_t TheHTM = adc/2.;
  HTMcut = TheHTM;
  return;
}

void L1AlgoFactory::ETTVal(Float_t& ETTcut) {

  Float_t adc = gt_->RankETT;
  Float_t TheETT = adc/2.;
  ETTcut = TheETT;
  return;
}

void L1AlgoFactory::ETMVal_NoQCD(Float_t& ETMcut ) {

  Float_t adc = gt_->RankETM;
  Float_t TheETM = adc/2.;

  Float_t ETM_Phi = gt_->PhiETM;

  Int_t Nj = gt_ -> Njet ;
  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    if(gt_->Taujet[ue]) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if(pt < 52.) continue;

    Float_t phijet = gt_->Phijet[ue];

    if(correlateInPhi(phijet,ETM_Phi,3)) return;
  }

  ETMcut = TheETM;

  return;
}

void L1AlgoFactory::Mu_EGPt(Float_t& mucut, Float_t& EGcut, Bool_t isIsolated, Int_t qualmin) {

  Float_t muptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  for(Int_t imu=0; imu < Nmu; imu++) {   
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( qual < qualmin) continue;
    if(pt >= muptmax) muptmax = pt;
  }

  Float_t eleptmax = -10.;

  Int_t Nele = gt_ -> Nele;
  for(Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    if(isIsolated && !gt_ -> Isoel[ue]) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if(pt >= eleptmax) eleptmax = pt;
  }

  if(muptmax >= 0. && eleptmax >= 0.){
    mucut = muptmax;
    EGcut = eleptmax;
  }

  return;
}


void L1AlgoFactory::DoubleMu_EGPt(Float_t& mucut, Float_t& EGcut, Bool_t isMuHighQual ) {

  Float_t muptmax = -10.;
  Float_t second_muptmax = -10.;
  Float_t EGmax = -10.;

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 2) return;

  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if(qual < 4 && qual !=3 ) continue;
    if(isMuHighQual && qual < 4) continue;
    if(pt >= muptmax){
      second_muptmax = muptmax;
      muptmax = pt;
    }
    else if(pt >= second_muptmax) second_muptmax = pt;
  }

  Int_t Nele = gt_ -> Nele;
  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if (bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if (pt >= EGmax) EGmax = pt;
  }  // end loop over EM objects

  if(second_muptmax >= 0.){
    mucut = second_muptmax;
    EGcut = EGmax;
  }

  return;
}

void L1AlgoFactory::Mu_DoubleEGPt(Float_t& mucut, Float_t& EGcut ) {

  Float_t muptmax    = -10.;
  Float_t eleptmax1  = -10.;
  Float_t eleptmax2  = -10.;
  Float_t ele1Phimax = -1000.;
  Float_t ele1Etamax = -1000.;

  Int_t Nmu = gmt_ -> N;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if(qual < 4) continue;
    if(pt >= muptmax) muptmax = pt; 
  }

  Int_t Nele = gt_ -> Nele;
  if(Nele < 2) return;

  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    Float_t phi = gt_ -> Phiel[ue];    // the rank of the electron
    Float_t eta = gt_ -> Etael[ue];    // the rank of the electron

    if(fabs(pt-eleptmax1) < 0.001 && fabs(phi-ele1Phimax) < 0.001 && fabs(eta-ele1Etamax) < 0.001) continue; //to avoid double counting in noniso/relaxiso lists

    if(pt >= eleptmax1){
      eleptmax2 = eleptmax1;
      eleptmax1 = pt;
      ele1Phimax = phi;
      ele1Etamax = eta;
    }
    else if(pt >= eleptmax2) eleptmax2 = pt;
  }

  if(muptmax >= 0. && eleptmax2 >= 0.){
    mucut = muptmax;
    EGcut = eleptmax2;
  }

  return;
}

void L1AlgoFactory::Mu_DoubleJetCentralPt(Float_t& mucut, Float_t& jetcut) {

  Float_t muptmax = -10.;
  Float_t jetptmax1 = -10.;
  Float_t jetptmax2 = -10.;

  Int_t Nmu = gmt_ -> N;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < 4) continue;
    if (pt >= muptmax) muptmax = pt;
  }

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 2) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if (isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= jetptmax1){
      jetptmax2 = jetptmax1;
      jetptmax1 = pt;
    }
    else if(pt >= jetptmax2) jetptmax2 = pt;
  }

  if(muptmax >= 0. && jetptmax2 >= 0.){
    mucut = muptmax;
    jetcut = jetptmax2;
  }

  return;
}

void L1AlgoFactory::Muer_JetCentralPt(Float_t& mucut, Float_t& jetcut) {

  Float_t muptmax = -10.;
  Float_t jetptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( qual < 4) continue;
    Float_t eta = gmt_  -> Eta[imu];        
    if(fabs(eta) > 2.1) continue;

    if (pt >= muptmax) muptmax = pt;
  }

  Int_t Nj = gt_ -> Njet ;
  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if (isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if (pt >= jetptmax) jetptmax = pt;
  }

  if(muptmax >= 0. && jetptmax >= 0.){
    mucut = muptmax;
    jetcut = jetptmax;
  }

  return;
}

void L1AlgoFactory::Mu_JetCentral_deltaPt(Float_t& mucut, Float_t& jetcut) {

  Float_t muptmax = -10.;
  Float_t jetptmax = -10.;
  Bool_t correlate = false;

  Int_t Nmu = gmt_ -> N;
  Int_t Nj = gt_ -> Njet;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < 4) continue;

    Float_t phimu = gmt_ -> Phi[imu];
    Int_t iphi_mu = phiINjetCoord(phimu);
    Float_t etamu = gmt_ -> Eta[imu];
    Int_t ieta_mu = etaINjetCoord(etamu);

    for (Int_t ue=0; ue < Nj; ue++) {
      Int_t bxj = gt_ -> Bxjet[ue];        		
      if (bxj != 0) continue;
      Bool_t isFwdJet = gt_ -> Fwdjet[ue];
      if (isFwdJet) continue;
      if(NOTauInJets && gt_->Taujet[ue]) continue;
      if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

      Float_t rank = gt_ -> Rankjet[ue];
      Float_t ptj = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
      Float_t phijet = gt_ -> Phijet[ue];
      Int_t iphi_jet = (int)phijet;
      Float_t etajet = gt_ -> Etajet[ue];
      Int_t ieta_jet = (int)etajet;

      if(pt < mucut || ptj < jetcut) continue;

      Bool_t corr = correlateInPhi(iphi_jet, iphi_mu, 2) && correlateInEta(ieta_jet, ieta_mu, 2);
      if(corr){
	correlate = true;
	if(pt >= muptmax) muptmax = pt;
	if(ptj >= jetptmax) jetptmax = ptj;
      }
    }
  }

  if(correlate){
    mucut = muptmax;
    jetcut = jetptmax;
  }

  return;
}

void L1AlgoFactory::Mu_HTTPt(Float_t& mucut, Float_t& HTcut ) {

  Float_t muptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( qual < 4) continue;
    if(pt >= muptmax) muptmax = pt;
  }

  Float_t adc = gt_ -> RankHTT ;
  Float_t TheHTT = adc/2.;

  if(muptmax >= 0.){
    mucut = muptmax;
    HTcut = TheHTT;
  }

  return;
}

void L1AlgoFactory::Muer_ETMPt(Float_t& mucut, Float_t& ETMcut ) {

  Float_t muptmax = -10.;

  Int_t Nmu = gmt_ -> N;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( qual < 4) continue;
    Float_t eta = gmt_  -> Eta[imu];        
    if(fabs(eta) > 2.1) continue;
    if(pt >= muptmax) muptmax = pt;
  }

  Float_t adc = gt_ -> RankETM ;
  Float_t TheETM = adc/2.;

  if(muptmax >= 0.){
    mucut = muptmax;
    ETMcut = TheETM;
  }

  return;
}

void L1AlgoFactory::SingleEG_Eta2p1_HTTPt(Float_t& egcut, Float_t& HTTcut, Bool_t isIsolated) {

  Float_t eleptmax = -10.;

  Int_t Nele = gt_ -> Nele;
  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Bool_t iso = gt_ -> Isoel[ue];
    if(isIsolated && !iso) continue;
    Float_t eta = gt_ -> Etael[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
    Float_t pt = gt_->Rankel[ue];
    if(pt >= eleptmax) eleptmax = pt;
  }

  Float_t adc = gt_ -> RankHTT ;
  Float_t TheHTT = adc/2.;

  if(eleptmax >= 0.){
    egcut = eleptmax;
    HTTcut = TheHTT;
  }

  return;
}

void L1AlgoFactory::EG_FwdJetPt(Float_t& EGcut, Float_t& FWcut) {

  Float_t eleptmax = -10.;
  Float_t jetptmax = -10.;

  Int_t Nele = gt_ -> Nele;
  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if(pt >= eleptmax) eleptmax = pt;
  }

  Int_t Nj = gt_ -> Njet ;
  for (Int_t ue=0; ue < Nj; ue++) {        
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(!isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if (pt >= jetptmax) jetptmax = pt;
  }

  if(eleptmax >= 0. && jetptmax >= 0.){
    EGcut = eleptmax;
    FWcut = jetptmax;
  }

  return;
}

void L1AlgoFactory::EG_DoubleJetCentralPt(Float_t& EGcut, Float_t& jetcut) {

  Float_t eleptmax = -10.;
  Float_t jetptmax1 = -10.;
  Float_t jetptmax2 = -10.;

  Int_t Nele = gt_ -> Nele;
  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if(pt >= eleptmax) eleptmax = pt; 
  }  // end loop over EM objects

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 2) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if (isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= jetptmax1){
      jetptmax2 = jetptmax1;
      jetptmax1 = pt;
    }
    else if(pt >= jetptmax2) jetptmax2 = pt;
  }

  if(eleptmax >= 0. && jetptmax2 >= 0.){
    EGcut = eleptmax;
    jetcut = jetptmax2;
  }

  return;
}

void L1AlgoFactory::EGer_TripleJetCentralPt(Float_t& EGcut, Float_t& jetcut) {

  Float_t eleptmax = -10.;
  Float_t elemaxeta = -10.;
  Float_t jetptmax1 = -10.;
  Float_t jetptmax2 = -10.;
  Float_t jetptmax3 = -10.;

  Int_t Nele = gt_ -> Nele;
  for (Int_t ue=0; ue < Nele; ue++){
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Float_t eta = gt_ -> Etael[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    if(pt >= eleptmax){
      eleptmax = pt; 
      elemaxeta = gt_->Etael[ue];
    }
  }  // end loop over EM objects

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 3) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if (isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    Float_t jeteta = gt_->Etajet[ue];
    if(jeteta == elemaxeta) continue;   //both are binned with the same binning

    if(pt >= jetptmax1){
      jetptmax3 = jetptmax2;
      jetptmax2 = jetptmax1;
      jetptmax1 = pt;
    }
    else if(pt >= jetptmax2){
      jetptmax3 = jetptmax2;
      jetptmax2 = pt;
    }
    else if(pt >= jetptmax3) jetptmax3 = pt;

  }

  if(eleptmax >= 0. && jetptmax3 >= 0.){
    EGcut = eleptmax;
    jetcut = jetptmax3;
  }

  return;
}

void L1AlgoFactory::DoubleEG_HTPt(Float_t& EGcut, Float_t& HTcut) {

  Float_t eleptmax1  = -10.;
  Float_t eleptmax2  = -10.;
  Float_t ele1Phimax = -1000.;
  Float_t ele1Etamax = -1000.;

  Int_t Nele = gt_ -> Nele;
  if(Nele < 2) return;

  for(Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Float_t pt = gt_ -> Rankel[ue];    // the rank of the electron
    Float_t phi = gt_ -> Phiel[ue];    // the rank of the electron
    Float_t eta = gt_ -> Etael[ue];    // the rank of the electron

    if(fabs(pt-eleptmax1) < 0.001 && fabs(phi-ele1Phimax) < 0.001 && fabs(eta-ele1Etamax) < 0.001) continue; //to avoid double counting in noniso/relaxiso lists

    if(pt >= eleptmax1){
      eleptmax2 = eleptmax1;
      eleptmax1 = pt;
      ele1Phimax = phi;
      ele1Etamax = eta;
    }
    else if(pt >= eleptmax2) eleptmax2 = pt;
  }

  Float_t adc = gt_ -> RankHTT ;
  Float_t TheHTT = adc / 2. ;

  if(eleptmax2 >= 0.){
    EGcut = eleptmax2;
    HTcut = TheHTT;
  }

  return;
}

void L1AlgoFactory::Jet_MuOpen_Mu_dPhiMuMu1Pt(Float_t& jetcut, Float_t& mucut) {

  //Find the highest pt jet with deltaphi condition
  Float_t jetptmax = -10.;
  Int_t Nj = gt_ -> Njet ;
  Int_t Nmu = gmt_ -> N;
  for(Int_t ue=0; ue < Nj; ue++){
    Int_t bxjet = gt_ -> Bxjet[ue];        		
    if(bxjet != 0) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if(pt < jetptmax) continue;
    Float_t phijet = gt_->Phijet[ue];

    Bool_t corr = false;

    for(Int_t imu=0; imu < Nmu; imu++){
      Int_t bx = gmt_ -> CandBx[imu];		
      if (bx != 0) continue;
      Int_t qual = gmt_ -> Qual[imu];        
      if(qual < 5 && qual != 3 ) continue; //The muon can be of lower quality
      if(gmt_->Pt[imu] < 0.) continue;
      Float_t muphi = phiINjetCoord(gmt_->Phi[imu]);
      if(fabs(muphi-phijet) < 3.) corr = true;
    }

    if(corr) jetptmax = pt;
  }

  //Loop over the muons list twice and save all pairs that satisfy the deltaphi veto
  Bool_t corr = false;
  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;

  std::vector<std::pair<Float_t,Float_t> > muonPairs;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if (bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if ( qual < 4) continue;   //one muon has SingleMu quality
    Float_t phi1 = gmt_->Phi[imu];        

    for (Int_t imu2=0; imu2 < Nmu; imu2++) {
      if (imu2 == imu) continue;
      Int_t bx2 = gmt_ -> CandBx[imu2];		
      if (bx2 != 0) continue;
      Float_t pt2 = gmt_ -> Pt[imu2];			
      Int_t qual2 = gmt_ -> Qual[imu2];        
      if(qual2 < 5 && qual != 3 ) continue; //The other muon can be of lower quality
      Float_t phi2 = gmt_->Phi[imu2];        

      Float_t dphi = phi1 - phi2; //Should get the binning, but for GMT is quite fine
      if(fabs(dphi) > 1.){
	corr = true;
	muonPairs.push_back(std::pair<Float_t,Float_t>(pt,pt2));
      }

    }
  }

  //Select the muon pair in which one of the two muons is the highest pt muon satisfying the deltaphi veto
  if(corr){
    std::vector<std::pair<Float_t,Float_t> >::const_iterator muonPairIt  = muonPairs.begin();
    std::vector<std::pair<Float_t,Float_t> >::const_iterator muonPairEnd = muonPairs.end();
    for(; muonPairIt != muonPairEnd; ++muonPairIt){
      Float_t pt1 = muonPairIt->first;
      Float_t pt2 = muonPairIt->second;
      
      if(pt1 > maxpt1 || (fabs(maxpt1-pt1)<10E-2 && pt2>maxpt2) ) {
	maxpt1 = pt1;
	maxpt2 = pt2;
      }
    }
  }

  Float_t maxptmu = maxpt1 > maxpt2 ? maxpt1 : maxpt2; //only the highest pt muon counts for the correlation, the second muon is Open

  if(jetptmax > 0. && maxptmu > 0.){
    jetcut = jetptmax;
    mucut = maxptmu;
  }

  return;
}

void L1AlgoFactory::Jet_MuOpen_EG_dPhiMuEG1Pt(Float_t& jetcut, Float_t& egcut){

  //Find the highest pt jet with deltaphi condition with MuOpen
  Float_t jetptmax = -10.;
  Int_t Nj = gt_ -> Njet ;
  Int_t Nmu = gmt_ -> N;
  for(Int_t ue=0; ue < Nj; ue++){
    Int_t bxjet = gt_ -> Bxjet[ue];        		
    if(bxjet != 0) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if(pt < jetptmax) continue;
    Float_t phijet = gt_->Phijet[ue];

    Bool_t corr = false;

    //Loop over all muons to check if a MuOpen is within 2 calo phi bins
    for(Int_t imu=0; imu < Nmu; imu++){
      Int_t bx = gmt_ -> CandBx[imu];		
      if (bx != 0) continue;
      Int_t qual = gmt_ -> Qual[imu];        
      if(qual < 5 && qual != 3) continue; //The muon can be of lower quality
      if(gmt_->Pt[imu] < 0.) continue;
      Float_t muphi = phiINjetCoord(gmt_->Phi[imu]);
      if(fabs(muphi-phijet) < 3.) corr = true;
    }

    if(corr) jetptmax = pt;
  }

  //Loop over electrons, and save all the electrons which satisfy the deltaphi veto with a MuOpen
  Int_t Nele = gt_ -> Nele;
  Float_t maxptEG = -10.;
  for(Int_t ue=0; ue < Nele; ue++){     
    Int_t bxele = gt_->Bxel[ue];        		
    if(bxele != 0) continue;
    Float_t pt = gt_->Rankel[ue]; // the rank of the electron
    Float_t EGphi = gt_->Phiel[ue];
    if(pt < maxptEG) continue;

    //Check the deltaphi veto with any muon
    Bool_t corr = false;
    for(Int_t imu=0; imu < Nmu; imu++) {
      Int_t bxmu = gmt_ -> CandBx[imu];		
      if(bxmu != 0) continue;
      Int_t qual = gmt_ -> Qual[imu];        
      if(qual < 5 && qual != 3) continue;
      if(gmt_->Pt[imu] < 0.) continue;
      Float_t muphi = phiINjetCoord(gmt_->Phi[imu]);

      if(fabs(muphi-EGphi) > 3.) corr = true;
    }

    if(corr) maxptEG = pt;
  }

  if(jetptmax > 0. && maxptEG > 0.){
    jetcut = jetptmax;
    egcut = maxptEG;
  }

  return;
}

void L1AlgoFactory::DoubleJetCentral_ETMPt(Float_t& jetcut1, Float_t& jetcut2, Float_t& ETMcut){

  Float_t jetptmax1 = -10.;
  Float_t jetptmax2 = -10.;

  Float_t adc = gt_ -> RankETM ;
  Float_t TheETM = adc / 2. ;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 2) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;
    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= jetptmax1){
      jetptmax2 = jetptmax1;
      jetptmax1 = pt;
    }
    else if(pt >= jetptmax2) jetptmax2 = pt;
  }       

  if(jetptmax2 >= 0.){
    jetcut1 = jetptmax1;
    jetcut2 = jetptmax2;
    ETMcut = TheETM;
  }

  return;
}

void L1AlgoFactory::Muer_TauJetEta2p17Pt(Float_t& mucut, Float_t& taucut) {

  Float_t maxptmu  = -10.;
  Float_t maxpttau = -10.;

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 1) return;
  for (Int_t imu=0; imu < Nmu; imu++) {
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Int_t qual = gmt_ -> Qual[imu];        
    if( qual < 4) continue;
    Float_t eta = gmt_  -> Eta[imu];        
    if(fabs(eta) > 2.1) continue;
    if(pt >= maxptmu) maxptmu = pt;
  }

  Int_t Nj = gt_ -> Njet ;
  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue; 
    Bool_t isTauJet = gt_ -> Taujet[ue];
    if(!isTauJet) continue;
    Float_t rank = gt_ -> Rankjet[ue];    // the rank of the electron
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    Float_t eta = gt_ -> Etajet[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16

    if(pt >= maxpttau) maxpttau = pt;
  }

  if(maxptmu >= 0.){
    mucut  = maxptmu;
    taucut = maxpttau;
  }

  return;
}

void L1AlgoFactory::IsoEGer_TauJetEta2p17Pt(Float_t& egcut, Float_t& taucut) {

  Float_t eleptmax  = -10.;
  Float_t eleetamax = -999.;
  Float_t maxpttau  = -10.;

  Int_t Nele = gt_ -> Nele;
  if(Nele < 1) return;
  for (Int_t ue=0; ue < Nele; ue++) {
    Int_t bx = gt_ -> Bxel[ue];        		
    if(bx != 0) continue;
    Bool_t iso = gt_ -> Isoel[ue];
    if(!iso) continue;
    Float_t eta = gt_ -> Etael[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16
    Float_t pt = gt_->Rankel[ue];
    if(pt >= eleptmax){
      eleptmax = pt;
      eleetamax = eta;
    }
  }

  Int_t Nj = gt_ -> Njet ;
  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue; 
    Bool_t isTauJet = gt_ -> Taujet[ue];
    if(!isTauJet) continue;
    Float_t rank = gt_ -> Rankjet[ue];    // the rank of the jet
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    Float_t eta = gt_ -> Etajet[ue];
    if(eta < 4.5 || eta > 16.5) continue;  // eta = 5 - 16

    if(fabs(eta-eleetamax) < 2) continue;

    if(pt >= maxpttau) maxpttau = pt;
  }

  if(eleptmax >= 0.){
    egcut  = eleptmax;
    taucut = maxpttau;
  }

  return;
}

void L1AlgoFactory::QuadJetCentral_TauJetPt(Float_t& jetcut, Float_t& taucut){

  Float_t jet1ptmax = -10.;
  Float_t jet2ptmax = -10.;
  Float_t jet3ptmax = -10.;
  Float_t jet4ptmax = -10.;
  Float_t maxpttau  = -10.;

  Int_t Nj = gt_ -> Njet ;
  if(Nj < 5) return;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);

    if(pt >= jet1ptmax)
      {
	jet4ptmax = jet3ptmax;
	jet3ptmax = jet2ptmax;
	jet2ptmax = jet1ptmax;
	jet1ptmax = pt;
      }
    else if(pt >= jet2ptmax){
      jet4ptmax = jet3ptmax;
      jet3ptmax = jet2ptmax;
      jet2ptmax = pt;
    }
    else if(pt >= jet3ptmax){
      jet4ptmax = jet3ptmax;
      jet3ptmax = pt;
    }
    else if(pt >= jet4ptmax) jet4ptmax = pt;
  }

  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue; 
    Bool_t isTauJet = gt_ -> Taujet[ue];
    if(!isTauJet) continue;
    Float_t rank = gt_ -> Rankjet[ue];    // the rank of the jet
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if(pt >= maxpttau) maxpttau = pt;
  }

  if(jet4ptmax >= 0. && maxpttau >= 0.){
    jetcut = jet4ptmax;
    taucut = maxpttau;
  }

  return;
}

void L1AlgoFactory::DoubleJetC_deltaPhi7_HTTPt(Float_t& jetcut, Float_t& httcut){

  Float_t adc = gt_->RankHTT;
  Float_t TheHTT = adc/2.;
  httcut = TheHTT;

  Int_t Nj = gt_->Njet;
  if(Nj < 2) return;

  Float_t maxpt1 = -10.;
  Float_t maxpt2 = -10.;
  Bool_t corr = false;
  std::vector<std::pair<Float_t,Float_t> > jetPairs;

  for (Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if (bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if (isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    Float_t phi1 = gt_ -> Phijet[ue];

    for(Int_t ve=0; ve < Nj; ve++) {
      if(ve == ue) continue;
      Int_t bx2 = gt_->Bxjet[ve];        		
      if(bx2!= 0) continue;
      Bool_t isFwdJet2 = gt_->Fwdjet[ve];
      if(isFwdJet2) continue;
      if(NOTauInJets && gt_->Taujet[ve]) continue;
      if(noHF && (gt_->Etajet[ve] < 5 || gt_->Etajet[ve] > 17)) continue;

      Float_t rank2 = gt_->Rankjet[ve];
      Float_t pt2 = rank2*4.;
      Float_t phi2 = gt_->Phijet[ve];

      if(correlateInPhi((int)phi1, (int)phi2, 7)){
	corr = true;
	jetPairs.push_back(std::pair<Float_t,Float_t>(pt,pt2));
      }

    }
  }

  if(corr){
    std::vector<std::pair<Float_t,Float_t> >::const_iterator jetPairIt  = jetPairs.begin();
    std::vector<std::pair<Float_t,Float_t> >::const_iterator jetPairEnd = jetPairs.end();
    for(; jetPairIt != jetPairEnd; ++jetPairIt) {
      Float_t pt1 = jetPairIt->first;
      Float_t pt2 = jetPairIt->second;
      
      if (pt1 > maxpt1 || (fabs(maxpt1-pt1)<10E-2 && pt2>maxpt2) ) 
	{
	  maxpt1 = pt1;
	  maxpt2 = pt2;
	}
    }
  }

  jetcut = maxpt2;

  return;
}

void L1AlgoFactory::Muer_ETM_HTTPt(Float_t& mucut, Float_t& ETMcut, Float_t& HTTcut){

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 1) return;

  Float_t ptmax = -10.;

  for(Int_t imu=0; imu < Nmu; imu++) { 
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Float_t eta = gmt_  -> Eta[imu];        
    if(fabs(eta) > 2.1) continue;
    Int_t qual = gmt_->Qual[imu];        
    if(qual < 4) continue;
    if(pt >= ptmax) ptmax = pt;
  }

  mucut = ptmax;

  Float_t adcETM = gt_->RankETM ;
  Float_t TheETM = adcETM/2.;
  ETMcut = TheETM;

  Float_t adcHTT = gt_->RankHTT ;
  Float_t TheHTT = adcHTT/2.;
  HTTcut = TheHTT;

  return;
}

void L1AlgoFactory::Muer_ETM_JetCPt(Float_t& mucut, Float_t& ETMcut, Float_t& jetcut){

  Int_t Nmu = gmt_ -> N;
  if(Nmu < 1) return;

  Float_t muptmax = -10.;

  for(Int_t imu=0; imu < Nmu; imu++) { 
    Int_t bx = gmt_ -> CandBx[imu];		
    if(bx != 0) continue;
    Float_t pt = gmt_ -> Pt[imu];			
    Float_t eta = gmt_  -> Eta[imu];        
    if(fabs(eta) > 2.1) continue;
    Int_t qual = gmt_->Qual[imu];        
    if(qual < 4) continue;
    if(pt >= muptmax) muptmax = pt;
  }

  mucut = muptmax;

  Float_t adcETM = gt_->RankETM ;
  Float_t TheETM = adcETM/2.;
  ETMcut = TheETM;

  Float_t jetptmax = -10.;
  Int_t Nj = gt_ -> Njet ;
  for(Int_t ue=0; ue < Nj; ue++) {
    Int_t bx = gt_ -> Bxjet[ue];        		
    if(bx != 0) continue;
    Bool_t isFwdJet = gt_ -> Fwdjet[ue];
    if(isFwdJet) continue;
    if(NOTauInJets && gt_->Taujet[ue]) continue;
    if(noHF && (gt_->Etajet[ue] < 5 || gt_->Etajet[ue] > 17)) continue;

    Float_t rank = gt_ -> Rankjet[ue];
    Float_t pt = CorrectedL1JetPtByGCTregions(gt_->Etajet[ue],rank*4.);
    if(pt >= jetptmax) jetptmax = pt;
  }

  jetcut = jetptmax;

  return;
}

#endif
