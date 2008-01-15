//
// $Id: LeptonLRCalc.cc,v 1.1 2008/01/07 11:48:27 lowette Exp $
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
double LeptonLRCalc::getCalIsoE(const Electron & electron) {
  return electron.getCaloIso();
}
double LeptonLRCalc::getCalIsoE(const Muon & muon) {
  return muon.getCaloIso();
}
double LeptonLRCalc::getCalIsoE(const Tau & tau) {
  //TODO
  return 0.;
}


// return the value for the TrIsoPt
double LeptonLRCalc::getTrIsoPt(const Electron & electron) {
  return electron.getTrackIso();
}
double LeptonLRCalc::getTrIsoPt(const Muon & muon) {
  return muon.getTrackIso();
}
double LeptonLRCalc::getTrIsoPt(const Tau & tau) {
  //TODO
  return 0.;
}


// return the value for the LepId
double LeptonLRCalc::getLepId(const Electron & electron) {
  return electron.getLeptonID();
}
double LeptonLRCalc::getLepId(const Muon & muon) {
  return muon.getLeptonID();
}
double LeptonLRCalc::getLepId(const Tau & tau) {
  //TODO
  return 1.;
}


// return the value for the LogPt
double LeptonLRCalc::getLogPt(const Electron & electron) {
  return log(electron.pt());
}
double LeptonLRCalc::getLogPt(const Muon & muon) {
  return log(muon.pt());
}
double LeptonLRCalc::getLogPt(const Tau & tau) {
  //  return log(tau.getJetTag()->jet().pt());
  return log(tau.pt());
}


// return the value for the JetIsoA
double LeptonLRCalc::getJetIsoA(const Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  return theJetIsoACalc_->calculate(electron, trackHandle, iEvent);
}
double LeptonLRCalc::getJetIsoA(const Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  return theJetIsoACalc_->calculate(muon, trackHandle, iEvent);
}
double LeptonLRCalc::getJetIsoA(const Tau & tau, const edm::Event & iEvent) {
  //TODO
  return 0.;
}


// return the value for the VtxSign
double LeptonLRCalc::getVtxSign(const Electron & electron, const edm::Event & iEvent) {
  return theVtxSignCalc_->calculate(electron, iEvent);
}
double LeptonLRCalc::getVtxSign(const Muon & muon, const edm::Event & iEvent) {
  return theVtxSignCalc_->calculate(muon, iEvent);
}
double LeptonLRCalc::getVtxSign(const Tau & tau, const edm::Event & iEvent) {
  //TODO
  return 1.;
}


// return the LR value for the CalIsoE
double LeptonLRCalc::getCalIsoELR(double calIsoE, LeptonType theType) {
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
double LeptonLRCalc::getTrIsoPtLR(double trIsoPt, LeptonType theType) {
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
double LeptonLRCalc::getLepIdLR(double lepId, LeptonType theType) {
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
double LeptonLRCalc::getLogPtLR(double logPt, LeptonType theType) {
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
double LeptonLRCalc::getJetIsoALR(double jetIsoA, LeptonType theType) {
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
double LeptonLRCalc::getVtxSignLR(double vtxSign, LeptonType theType) {
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
void LeptonLRCalc::calcLRVars(Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  electron.setLRVarVal(std::pair<double, double>(this->getCalIsoE(electron), 0), 0);
  electron.setLRVarVal(std::pair<double, double>(this->getTrIsoPt(electron), 0), 1);
  electron.setLRVarVal(std::pair<double, double>(this->getLepId(electron), 0), 2);
  electron.setLRVarVal(std::pair<double, double>(this->getLogPt(electron), 0), 3);
  electron.setLRVarVal(std::pair<double, double>(this->getJetIsoA(electron, trackHandle, iEvent), 0), 4);
  electron.setLRVarVal(std::pair<double, double>(this->getVtxSign(electron, iEvent), 0), 5);
}


// do the LR calculations for an electron
void LeptonLRCalc::calcLRVals(Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  this->calcLRVars(electron, trackHandle, iEvent);
  // store the LR values
  LeptonType electronType = ElectronT;
  electron.setLRVarVal(std::pair<double, double>(electron.getLRVar(0), this->getCalIsoELR(electron.getLRVar(0), electronType)), 0);
  electron.setLRVarVal(std::pair<double, double>(electron.getLRVar(1), this->getTrIsoPtLR(electron.getLRVar(1), electronType)), 1);
  electron.setLRVarVal(std::pair<double, double>(electron.getLRVar(2), this->getLepIdLR(electron.getLRVar(2), electronType)), 2);
  electron.setLRVarVal(std::pair<double, double>(electron.getLRVar(3), this->getLogPtLR(electron.getLRVar(3), electronType)), 3);
  electron.setLRVarVal(std::pair<double, double>(electron.getLRVar(4), this->getJetIsoALR(electron.getLRVar(4), electronType)), 4);
  electron.setLRVarVal(std::pair<double, double>(electron.getLRVar(5), this->getVtxSignLR(electron.getLRVar(5), electronType)), 5);
}


// calculate the combined likelihood
void LeptonLRCalc::calcLikelihood(Electron & electron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  this->calcLRVals(electron, trackHandle, iEvent);
  // combine through dumb product of the lr values
  double lrComb = 1;
  for (unsigned int i = 0; i < electron.getLRSize(); i++) {
    lrComb *= (electron.getLRVal(i) / electronFitMax_[i]);
  }
  // combine via the product of s/b
//  double combSOverB = 1;
//  // lr = s/(s+b) -> (s+b)/s = 1/lr = 1+b/s -> (1-lr)/lr = b/s -> s/b = lr/(1-lr)
//  for (unsigned int i = 0; i < electron.getLRSize(); i++) {
//    combSOverB *= electron.getLRVal(i) / (1 - electron.getLRVal(i));
//  }
//  // lr = 1/[(s+b)/s] -> lr = 1/(1+1/(s/b))
//  double lrComb = 1 / (1 + 1/combSOverB);
  electron.setLRComb(lrComb);
}


// do the LR variable calculations for a muon
void LeptonLRCalc::calcLRVars(Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  muon.setLRVarVal(std::pair<double, double>(this->getCalIsoE(muon), 0), 0);
  muon.setLRVarVal(std::pair<double, double>(this->getTrIsoPt(muon), 0), 1);
  muon.setLRVarVal(std::pair<double, double>(this->getLepId(muon), 0), 2);
  muon.setLRVarVal(std::pair<double, double>(this->getLogPt(muon), 0), 3);
  muon.setLRVarVal(std::pair<double, double>(this->getJetIsoA(muon, trackHandle, iEvent), 0), 4);
  muon.setLRVarVal(std::pair<double, double>(this->getVtxSign(muon, iEvent), 0), 5);
}


// do the LR calculations for a muon
void LeptonLRCalc::calcLRVals(Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  this->calcLRVars(muon, trackHandle, iEvent);
  // store the LR values
  LeptonType muonType = MuonT;
  muon.setLRVarVal(std::pair<double, double>(muon.getLRVar(0), this->getCalIsoELR(muon.getLRVar(0), muonType)), 0);
  muon.setLRVarVal(std::pair<double, double>(muon.getLRVar(1), this->getTrIsoPtLR(muon.getLRVar(1), muonType)), 1);
  muon.setLRVarVal(std::pair<double, double>(muon.getLRVar(2), this->getLepIdLR(muon.getLRVar(2), muonType)), 2);
  muon.setLRVarVal(std::pair<double, double>(muon.getLRVar(3), this->getLogPtLR(muon.getLRVar(3), muonType)), 3);
  muon.setLRVarVal(std::pair<double, double>(muon.getLRVar(4), this->getJetIsoALR(muon.getLRVar(4), muonType)), 4);
  muon.setLRVarVal(std::pair<double, double>(muon.getLRVar(5), this->getVtxSignLR(muon.getLRVar(5), muonType)), 5);
}


// calculate the combined likelihood
void LeptonLRCalc::calcLikelihood(Muon & muon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent) {
  this->calcLRVals(muon, trackHandle, iEvent);
  // combine through dumb product of the lr values
  double lrComb = 1;
  for (unsigned int i = 0; i < muon.getLRSize(); i++) {
    lrComb *= (muon.getLRVal(i) / muonFitMax_[i]);
  }
//  // combine via the product of s/b
//  double combSOverB = 1;
//  // lr = s/(s+b) -> (s+b)/s = 1/lr = 1+b/s -> (1-lr)/lr = b/s -> s/b = lr/(1-lr)
//  for (unsigned int i = 0; i < muon.getLRSize(); i++) {
//    combSOverB *= muon.getLRVal(i) / (1 - muon.getLRVal(i));
//  }
//  // lr = 1/[(s+b)/s] -> lr = 1/(1+1/(s/b))
//  double lrComb = 1 / (1 + 1/combSOverB);
  muon.setLRComb(lrComb);
}

// do the LR variable calculations for a tau
void LeptonLRCalc::calcLRVars(Tau & tau, const edm::Event & iEvent) {
  tau.setLRVarVal(std::pair<double, double>(this->getCalIsoE(tau), 0), 0);
  tau.setLRVarVal(std::pair<double, double>(this->getTrIsoPt(tau), 0), 1);
  tau.setLRVarVal(std::pair<double, double>(this->getLepId(tau), 0), 2);
  tau.setLRVarVal(std::pair<double, double>(this->getLogPt(tau), 0), 3);
  tau.setLRVarVal(std::pair<double, double>(this->getJetIsoA(tau, iEvent), 0), 4);
  tau.setLRVarVal(std::pair<double, double>(this->getVtxSign(tau, iEvent), 0), 5);
}

// do the LR calculations for a tau
void LeptonLRCalc::calcLRVals(Tau & tau, const edm::Event & iEvent) {
  this->calcLRVars(tau, iEvent);
  // store the LR values
  LeptonType tauType = TauT;
  tau.setLRVarVal(std::pair<double, double>(tau.getLRVar(0), this->getCalIsoELR(tau.getLRVar(0), tauType)), 0);
  tau.setLRVarVal(std::pair<double, double>(tau.getLRVar(1), this->getTrIsoPtLR(tau.getLRVar(1), tauType)), 1);
  tau.setLRVarVal(std::pair<double, double>(tau.getLRVar(2), this->getLepIdLR(tau.getLRVar(2), tauType)), 2);
  tau.setLRVarVal(std::pair<double, double>(tau.getLRVar(3), this->getLogPtLR(tau.getLRVar(3), tauType)), 3);
  tau.setLRVarVal(std::pair<double, double>(tau.getLRVar(4), this->getJetIsoALR(tau.getLRVar(4), tauType)), 4);
  tau.setLRVarVal(std::pair<double, double>(tau.getLRVar(5), this->getVtxSignLR(tau.getLRVar(5), tauType)), 5);
}

// calculate the combined likelihood
void LeptonLRCalc::calcLikelihood(Tau & tau, const edm::Event & iEvent) {
  this->calcLRVals(tau, iEvent);
  // combine through dumb product of the lr values
//  double lrComb;
//  for (unsigned int i = 0; i < tau.getLRSize(); i++) {
//    lrComb *= tau.getLRVal(i);
//  }
  // combine via the product of s/b
  double combSOverB = 1;
  // lr = s/(s+b) -> (s+b)/s = 1/lr = 1+b/s -> (1-lr)/lr = b/s -> s/b = lr/(1-lr)
  for (unsigned int i = 0; i < tau.getLRSize(); i++) {
    combSOverB *= tau.getLRVal(i) / (1 - tau.getLRVal(i));
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
  if (!electronFile) std::cout << "*** ERROR: fitFile " << electronLRFile_ << " not found. I will most likely crash now..." << std::endl;
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
  if (!muonFile) std::cout << "*** ERROR: fitFile " << muonLRFile_ << " not found. I will most likely crash now..." << std::endl;
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
  if (!tauFile) std::cout << "*** ERROR: fitFile " << tauLRFile_ << " not found. I will most likely crash now..." << std::endl;
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

