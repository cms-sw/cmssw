//
// $Id: LeptonLRCalc.cc,v 1.3 2008/01/21 16:26:20 lowette Exp $
//

#include "PhysicsTools/PatUtils/interface/LeptonLRCalc.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/PatUtils/interface/LeptonJetIsolationAngle.h"
#include "PhysicsTools/PatUtils/interface/LeptonVertexSignificance.h"

#include "TFile.h"
#include "TKey.h"
#include "TH1.h"
#include "TString.h"

// FIXME: get rid of this, use the messagelogger
#include <iostream>

//TODO: Tau methods are just fake placeholders obtained by cut-and-paste of electron/muon cases.
// It should be tuned and replaced by actual code.


using namespace pat;


// constructor with path; default should not be used
LeptonLRCalc::LeptonLRCalc(const edm::EventSetup & iSetup, std::string electronLRFile, std::string muonLRFile, std::string tauLRFile) {
  electronLRFile_ = electronLRFile;
  muonLRFile_ = muonLRFile;
  tauLRFile_ = tauLRFile;

  theJetIsoACalc_ = new LeptonJetIsolationAngle();
  theVtxSignCalc_ = new LeptonVertexSignificance(iSetup);

  fitsElectronRead_ = false;
  fitsMuonRead_ = false;
  fitsTauRead_ = false;
}


// destructor
LeptonLRCalc::~LeptonLRCalc() {
  delete theVtxSignCalc_;
  delete theJetIsoACalc_;
}


// return the value for the CalIsoE
double LeptonLRCalc::calIsoE(const Electron & electron) {
  return electron.caloIso();
}
double LeptonLRCalc::calIsoE(const Muon & muon) {
  return muon.caloIso();
}
double LeptonLRCalc::calIsoE(const Tau & tau) {
  //TODO
  return 0.;
}


// return the value for the TrIsoPt
double LeptonLRCalc::trIsoPt(const Electron & electron) {
  return electron.trackIso();
}
double LeptonLRCalc::trIsoPt(const Muon & muon) {
  return muon.trackIso();
}
double LeptonLRCalc::trIsoPt(const Tau & tau) {
  //TODO
  return 0.;
}


// return the value for the LepId
double LeptonLRCalc::lepId(const Electron & electron) {
  return electron.leptonID();
}
double LeptonLRCalc::lepId(const Muon & muon) {
  return muon.leptonID();
}
double LeptonLRCalc::lepId(const Tau & tau) {
  //TODO
  return 1.;
}


// return the value for the LogPt
double LeptonLRCalc::logPt(const Electron & electron) {
  return log(electron.pt());
}
double LeptonLRCalc::logPt(const Muon & muon) {
  return log(muon.pt());
}
double LeptonLRCalc::logPt(const Tau & tau) {
  //  return log(tau.jetTag()->jet().pt());
  return log(tau.pt());
}


// return the value for the JetIsoA
double LeptonLRCalc::jetIsoA(const Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  return theJetIsoACalc_->calculate(electron, trackHandle, iEvent);
}
double LeptonLRCalc::jetIsoA(const Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  return theJetIsoACalc_->calculate(muon, trackHandle, iEvent);
}
double LeptonLRCalc::jetIsoA(const Tau & tau, const edm::Event & iEvent) {
  //TODO
  return 0.;
}


// return the value for the VtxSign
double LeptonLRCalc::vtxSign(const Electron & electron, const edm::Event & iEvent) {
  return theVtxSignCalc_->calculate(electron, iEvent);
}
double LeptonLRCalc::vtxSign(const Muon & muon, const edm::Event & iEvent) {
  return theVtxSignCalc_->calculate(muon, iEvent);
}
double LeptonLRCalc::vtxSign(const Tau & tau, const edm::Event & iEvent) {
  //TODO
  return 1.;
}


// return the LR value for the CalIsoE
double LeptonLRCalc::calIsoELR(double calIsoE, const LeptonType & theType) {
  double lrVal = 1;
  if (theType == ElectronT) {
    if (!fitsElectronRead_) readFitsElectron();
    lrVal = electronCalIsoEFit.Eval(calIsoE);
  } else if (theType == MuonT) {
    if (!fitsMuonRead_) readFitsMuon();
    lrVal = muonCalIsoEFit.Eval(calIsoE);
  }
  //TODO: add tau case.
  return lrVal;
}


// return the LR value for the TrIsoPt
double LeptonLRCalc::trIsoPtLR(double trIsoPt, const LeptonType & theType) {
  double lrVal = 1;
  if (theType == ElectronT) {
    if (!fitsElectronRead_) readFitsElectron();
// TEMP HACK TO USE log(sumPt) FIT WHILE VARIABLE CHANGED TO sum(pT)
trIsoPt <= 0.01 ? trIsoPt = -1 : trIsoPt = log(trIsoPt);
    lrVal = electronTrIsoPtFit.Eval(trIsoPt);
  } else if (theType == MuonT) {
    if (!fitsMuonRead_) readFitsMuon();
    lrVal = muonTrIsoPtFit.Eval(trIsoPt);
  }
  //TODO: add tau case.
  return lrVal;
}


// return the LR value for the LeptonID
double LeptonLRCalc::lepIdLR(double lepId, const LeptonType & theType) {
  double lrVal = 1;
  if (theType == ElectronT) {
    if (!fitsElectronRead_) readFitsElectron();
    lrVal = electronLepIdFit.Eval(lepId);
  } else if (theType == MuonT) {
    if (!fitsMuonRead_) readFitsMuon();
    lrVal = muonLepIdFit.Eval(lepId);
  }
  //TODO: add tau case.
  return lrVal;
}


// return the LR value for the LogPt
double LeptonLRCalc::logPtLR(double logPt, const LeptonType & theType) {
  double lrVal = 1;
  if (theType == ElectronT) {
    if (!fitsElectronRead_) readFitsElectron();
    lrVal = electronLogPtFit.Eval(logPt);
  } else if (theType == MuonT) {
    if (!fitsMuonRead_) readFitsMuon();
    lrVal = muonLogPtFit.Eval(logPt);
  }
  //TODO: add tau case.
  return lrVal;
}


// return the LR value for the JetIsoA
double LeptonLRCalc::jetIsoALR(double jetIsoA, const LeptonType & theType) {
  double lrVal = 1;
  if (theType == ElectronT) {
    if (!fitsElectronRead_) readFitsElectron();
    lrVal = electronJetIsoAFit.Eval(jetIsoA);
  } else if (theType == MuonT) {
    if (!fitsMuonRead_) readFitsMuon();
    lrVal = muonJetIsoAFit.Eval(jetIsoA);
  }
  //TODO: add tau case.
  return lrVal;
}


// return the LR value for the VtxSign
double LeptonLRCalc::vtxSignLR(double vtxSign, const LeptonType & theType) {
  double lrVal = 1;
  if (theType == ElectronT) {
    if (!fitsElectronRead_) readFitsElectron();
    lrVal = electronVtxSignFit.Eval(vtxSign);
  } else if (theType == MuonT) {
    if (!fitsMuonRead_) readFitsMuon();
    lrVal = muonVtxSignFit.Eval(vtxSign);
  }
  //TODO: add tau case.
  return lrVal;
}


// do the LR variable calculations for an electron
void LeptonLRCalc::calcLRVars(Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  electron.setLRVarVal(std::pair<double, double>(this->calIsoE(electron), 0), 0);
  electron.setLRVarVal(std::pair<double, double>(this->trIsoPt(electron), 0), 1);
  electron.setLRVarVal(std::pair<double, double>(this->lepId(electron), 0), 2);
  electron.setLRVarVal(std::pair<double, double>(this->logPt(electron), 0), 3);
  electron.setLRVarVal(std::pair<double, double>(this->jetIsoA(electron, trackHandle, iEvent), 0), 4);
  electron.setLRVarVal(std::pair<double, double>(this->vtxSign(electron, iEvent), 0), 5);
}


// do the LR calculations for an electron
void LeptonLRCalc::calcLRVals(Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  this->calcLRVars(electron, trackHandle, iEvent);
  // store the LR values
  LeptonType electronType = ElectronT;
  electron.setLRVarVal(std::pair<double, double>(electron.lrVar(0), this->calIsoELR(electron.lrVar(0), electronType)), 0);
  electron.setLRVarVal(std::pair<double, double>(electron.lrVar(1), this->trIsoPtLR(electron.lrVar(1), electronType)), 1);
  electron.setLRVarVal(std::pair<double, double>(electron.lrVar(2), this->lepIdLR(electron.lrVar(2), electronType)), 2);
  electron.setLRVarVal(std::pair<double, double>(electron.lrVar(3), this->logPtLR(electron.lrVar(3), electronType)), 3);
  electron.setLRVarVal(std::pair<double, double>(electron.lrVar(4), this->jetIsoALR(electron.lrVar(4), electronType)), 4);
  electron.setLRVarVal(std::pair<double, double>(electron.lrVar(5), this->vtxSignLR(electron.lrVar(5), electronType)), 5);
}


// calculate the combined likelihood
void LeptonLRCalc::calcLikelihood(Electron & electron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  this->calcLRVals(electron, trackHandle, iEvent);
  // combine through dumb product of the lr values
  double lrComb = 1;
  for (unsigned int i = 0; i < electron.lrSize(); i++) {
    lrComb *= (electron.lrVal(i) / electronFitMax_[i]);
  }
  // combine via the product of s/b
//  double combSOverB = 1;
//  // lr = s/(s+b) -> (s+b)/s = 1/lr = 1+b/s -> (1-lr)/lr = b/s -> s/b = lr/(1-lr)
//  for (unsigned int i = 0; i < electron.lrSize(); i++) {
//    combSOverB *= electron.lrVal(i) / (1 - electron.lrVal(i));
//  }
//  // lr = 1/[(s+b)/s] -> lr = 1/(1+1/(s/b))
//  double lrComb = 1 / (1 + 1/combSOverB);
  electron.setLRComb(lrComb);
}


// do the LR variable calculations for a muon
void LeptonLRCalc::calcLRVars(Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  muon.setLRVarVal(std::pair<double, double>(this->calIsoE(muon), 0), 0);
  muon.setLRVarVal(std::pair<double, double>(this->trIsoPt(muon), 0), 1);
  muon.setLRVarVal(std::pair<double, double>(this->lepId(muon), 0), 2);
  muon.setLRVarVal(std::pair<double, double>(this->logPt(muon), 0), 3);
  muon.setLRVarVal(std::pair<double, double>(this->jetIsoA(muon, trackHandle, iEvent), 0), 4);
  muon.setLRVarVal(std::pair<double, double>(this->vtxSign(muon, iEvent), 0), 5);
}


// do the LR calculations for a muon
void LeptonLRCalc::calcLRVals(Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  this->calcLRVars(muon, trackHandle, iEvent);
  // store the LR values
  LeptonType muonType = MuonT;
  muon.setLRVarVal(std::pair<double, double>(muon.lrVar(0), this->calIsoELR(muon.lrVar(0), muonType)), 0);
  muon.setLRVarVal(std::pair<double, double>(muon.lrVar(1), this->trIsoPtLR(muon.lrVar(1), muonType)), 1);
  muon.setLRVarVal(std::pair<double, double>(muon.lrVar(2), this->lepIdLR(muon.lrVar(2), muonType)), 2);
  muon.setLRVarVal(std::pair<double, double>(muon.lrVar(3), this->logPtLR(muon.lrVar(3), muonType)), 3);
  muon.setLRVarVal(std::pair<double, double>(muon.lrVar(4), this->jetIsoALR(muon.lrVar(4), muonType)), 4);
  muon.setLRVarVal(std::pair<double, double>(muon.lrVar(5), this->vtxSignLR(muon.lrVar(5), muonType)), 5);
}


// calculate the combined likelihood
void LeptonLRCalc::calcLikelihood(Muon & muon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  this->calcLRVals(muon, trackHandle, iEvent);
  // combine through dumb product of the lr values
  double lrComb = 1;
  for (unsigned int i = 0; i < muon.lrSize(); i++) {
    lrComb *= (muon.lrVal(i) / muonFitMax_[i]);
  }
//  // combine via the product of s/b
//  double combSOverB = 1;
//  // lr = s/(s+b) -> (s+b)/s = 1/lr = 1+b/s -> (1-lr)/lr = b/s -> s/b = lr/(1-lr)
//  for (unsigned int i = 0; i < muon.lrSize(); i++) {
//    combSOverB *= muon.lrVal(i) / (1 - muon.lrVal(i));
//  }
//  // lr = 1/[(s+b)/s] -> lr = 1/(1+1/(s/b))
//  double lrComb = 1 / (1 + 1/combSOverB);
  muon.setLRComb(lrComb);
}

// do the LR variable calculations for a tau
void LeptonLRCalc::calcLRVars(Tau & tau, const edm::Event & iEvent) {
  tau.setLRVarVal(std::pair<double, double>(this->calIsoE(tau), 0), 0);
  tau.setLRVarVal(std::pair<double, double>(this->trIsoPt(tau), 0), 1);
  tau.setLRVarVal(std::pair<double, double>(this->lepId(tau), 0), 2);
  tau.setLRVarVal(std::pair<double, double>(this->logPt(tau), 0), 3);
  tau.setLRVarVal(std::pair<double, double>(this->jetIsoA(tau, iEvent), 0), 4);
  tau.setLRVarVal(std::pair<double, double>(this->vtxSign(tau, iEvent), 0), 5);
}

// do the LR calculations for a tau
void LeptonLRCalc::calcLRVals(Tau & tau, const edm::Event & iEvent) {
  this->calcLRVars(tau, iEvent);
  // store the LR values
  LeptonType tauType = TauT;
  tau.setLRVarVal(std::pair<double, double>(tau.lrVar(0), this->calIsoELR(tau.lrVar(0), tauType)), 0);
  tau.setLRVarVal(std::pair<double, double>(tau.lrVar(1), this->trIsoPtLR(tau.lrVar(1), tauType)), 1);
  tau.setLRVarVal(std::pair<double, double>(tau.lrVar(2), this->lepIdLR(tau.lrVar(2), tauType)), 2);
  tau.setLRVarVal(std::pair<double, double>(tau.lrVar(3), this->logPtLR(tau.lrVar(3), tauType)), 3);
  tau.setLRVarVal(std::pair<double, double>(tau.lrVar(4), this->jetIsoALR(tau.lrVar(4), tauType)), 4);
  tau.setLRVarVal(std::pair<double, double>(tau.lrVar(5), this->vtxSignLR(tau.lrVar(5), tauType)), 5);
}

// calculate the combined likelihood
void LeptonLRCalc::calcLikelihood(Tau & tau, const edm::Event & iEvent) {
  this->calcLRVals(tau, iEvent);
  // combine through dumb product of the lr values
//  double lrComb;
//  for (unsigned int i = 0; i < tau.lrSize(); i++) {
//    lrComb *= tau.lrVal(i);
//  }
  // combine via the product of s/b
  double combSOverB = 1;
  // lr = s/(s+b) -> (s+b)/s = 1/lr = 1+b/s -> (1-lr)/lr = b/s -> s/b = lr/(1-lr)
  for (unsigned int i = 0; i < tau.lrSize(); i++) {
    combSOverB *= tau.lrVal(i) / (1 - tau.lrVal(i));
  }
  // lr = 1/[(s+b)/s] -> lr = 1/(1+1/(s/b))
  double lrComb = 1 / (1 + 1/combSOverB);
  tau.setLRComb(lrComb);
}

// read in electron fitfunctions
void LeptonLRCalc::readFitsElectron() {
  // Read in the fit functions from the rootfile
  TFile * electronFile = new TFile(TString(electronLRFile_));
  // FIXME: use messagelogger
  if (!electronFile) edm::LogError("LeptonLRCalc") << "*** ERROR: fitFile " << electronLRFile_ << " not found. I will most likely crash now..." << std::endl;
  TH1 * electronLRHist1 = (TH1 *) electronFile->GetKey("myLRPlot10")->ReadObj();
  TH1 * electronLRHist2 = (TH1 *) electronFile->GetKey("myLRPlot20")->ReadObj();
  TH1 * electronLRHist3 = (TH1 *) electronFile->GetKey("myLRPlot30")->ReadObj();
  TH1 * electronLRHist4 = (TH1 *) electronFile->GetKey("myLRPlot40")->ReadObj();
  TH1 * electronLRHist5 = (TH1 *) electronFile->GetKey("myLRPlot50")->ReadObj();
  TH1 * electronLRHist6 = (TH1 *) electronFile->GetKey("myLRPlot60")->ReadObj();
  electronCalIsoEFit = *(electronLRHist1->GetFunction("myLRFit10"));
  electronTrIsoPtFit = *(electronLRHist2->GetFunction("myLRFit20"));
  electronLepIdFit   = *(electronLRHist3->GetFunction("myLRFit30"));
  electronLogPtFit   = *(electronLRHist4->GetFunction("myLRFit40"));
  electronJetIsoAFit = *(electronLRHist5->GetFunction("myLRFit50"));
  electronVtxSignFit = *(electronLRHist6->GetFunction("myLRFit60"));
  electronFitMax_[0] = electronCalIsoEFit.GetMaximum(-20, 20);
  electronFitMax_[1] = electronTrIsoPtFit.Eval(-1);
  electronFitMax_[2] = electronLepIdFit.Eval(1);
  electronFitMax_[3] = electronLogPtFit.GetParameter(0);
  electronFitMax_[4] = electronJetIsoAFit.GetParameter(0);
  electronFitMax_[5] = electronVtxSignFit.Eval(0);
  delete electronFile;
  fitsElectronRead_ = true;
}


// read in muon fitfunctions
void LeptonLRCalc::readFitsMuon() {
  // Read in the fit functions from the rootfile
  TFile * muonFile = new TFile(TString(muonLRFile_));
  // FIXME: use messagelogger
  if (!muonFile) edm::LogError("LeptonLRCalc") << "*** ERROR: fitFile " << muonLRFile_ << " not found. I will most likely crash now..." << std::endl;
  TH1 * muonLRHist1 = (TH1 *) muonFile->GetKey("myLRPlot11")->ReadObj();
  TH1 * muonLRHist2 = (TH1 *) muonFile->GetKey("myLRPlot21")->ReadObj();
  TH1 * muonLRHist3 = (TH1 *) muonFile->GetKey("myLRPlot31")->ReadObj();
  TH1 * muonLRHist4 = (TH1 *) muonFile->GetKey("myLRPlot41")->ReadObj();
  TH1 * muonLRHist5 = (TH1 *) muonFile->GetKey("myLRPlot51")->ReadObj();
  TH1 * muonLRHist6 = (TH1 *) muonFile->GetKey("myLRPlot61")->ReadObj();
  muonCalIsoEFit = *(muonLRHist1->GetFunction("myLRFit11"));
  muonTrIsoPtFit = *(muonLRHist2->GetFunction("myLRFit21"));
  muonLepIdFit   = *(muonLRHist3->GetFunction("myLRFit31"));
  muonLogPtFit   = *(muonLRHist4->GetFunction("myLRFit41"));
  muonJetIsoAFit = *(muonLRHist5->GetFunction("myLRFit51"));
  muonVtxSignFit = *(muonLRHist6->GetFunction("myLRFit61"));
  muonFitMax_[0] = muonCalIsoEFit.Eval(0);
  muonFitMax_[1] = muonTrIsoPtFit.Eval(-1);
  muonFitMax_[2] = muonLepIdFit.GetParameter(0);
  muonFitMax_[3] = muonLogPtFit.GetParameter(0);
  muonFitMax_[4] = muonJetIsoAFit.GetParameter(0);
  muonFitMax_[5] = muonVtxSignFit.Eval(0);
  delete muonFile;
  fitsMuonRead_ = true;
}

void LeptonLRCalc::readFitsTau() {
  // initialisation
  // Read in the fit functions from the rootfile
  TFile * tauFile = new TFile(TString(tauLRFile_));
  // FIXME: use messagelogger or throw exception
  if (!tauFile) edm::LogError("LeptonLRCalc") << "*** ERROR: fitFile " << tauLRFile_ << " not found. I will most likely crash now..." << std::endl;
  // FIXME: names and order of the fits
  TH1 * tauLRHist1 = (TH1 *) tauFile->GetKey("myLRPlot12")->ReadObj();
  TH1 * tauLRHist2 = (TH1 *) tauFile->GetKey("myLRPlot22")->ReadObj();
  TH1 * tauLRHist3 = (TH1 *) tauFile->GetKey("myLRPlot32")->ReadObj();
  TH1 * tauLRHist4 = (TH1 *) tauFile->GetKey("myLRPlot42")->ReadObj();
  TH1 * tauLRHist5 = (TH1 *) tauFile->GetKey("myLRPlot52")->ReadObj();
  TH1 * tauLRHist6 = (TH1 *) tauFile->GetKey("myLRPlot62")->ReadObj();
  tauCalIsoEFit = *(tauLRHist1->GetFunction("myLRFit12"));
  tauTrIsoPtFit = *(tauLRHist2->GetFunction("myLRFit22"));
  tauLepIdFit   = *(tauLRHist3->GetFunction("myLRFit32"));
  tauLogPtFit   = *(tauLRHist4->GetFunction("myLRFit42"));
  tauJetIsoAFit = *(tauLRHist5->GetFunction("myLRFit52"));
  tauVtxSignFit = *(tauLRHist6->GetFunction("myLRFit62"));
  delete tauFile;
  fitsTauRead_ = true;
}

