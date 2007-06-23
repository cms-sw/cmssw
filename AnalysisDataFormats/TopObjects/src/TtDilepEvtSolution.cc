#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"

TtDilepEvtSolution::TtDilepEvtSolution()
{
  bestSol	= false;
  WpDecay       = "NotDefined";
  WmDecay       = "NotDefined";
}


TtDilepEvtSolution::~TtDilepEvtSolution()
{
}

void TtDilepEvtSolution::setGenEvt(const TtGenEvent& genEvt)
{
  if( !genEvt.isFullLeptonic() ){
    throw edm::Exception( edm::errors::Configuration, "found genEvt which is not di-leptonic" );
  }
  genLepm = *(genEvt.lepton());
  genLepp = *(genEvt.leptonBar());
  genN    = *(genEvt.neutrino());
  genNbar = *(genEvt.neutrinoBar());
  genB    = *(genEvt.b());
  genBbar = *(genEvt.bBar());;
  genT    = *(genEvt.top());
  genTbar = *(genEvt.topBar());
  genWm   = *(genEvt.w());
  genWp   = *(genEvt.wBar());
}

void TtDilepEvtSolution::setBestSol(bool bs) { bestSol = bs; }

//SolutionMaker Method should call next methods
void TtDilepEvtSolution::setMuonLepp(TopMuon j) {muonLepp = j; WpDecay = "muon";}
void TtDilepEvtSolution::setMuonLepm(TopMuon j) {muonLepm = j; WmDecay = "muon";}
void TtDilepEvtSolution::setElectronLepp(TopElectron j) {elecLepp = j; WpDecay = "electron";}
void TtDilepEvtSolution::setElectronLepm(TopElectron j) {elecLepm = j; WmDecay = "electron";}
void TtDilepEvtSolution::setB(TopJet j) {jetB = j;}
void TtDilepEvtSolution::setBbar(TopJet j) {jetBbar = j;}
void TtDilepEvtSolution::setMET(TopMET j) {met = j;}



void TtDilepEvtSolution::setRecTopMass(double j) {topmass_ = j;}
void TtDilepEvtSolution::setRecWeightMax(double j) {weightmax_ = j;}

TopJetType TtDilepEvtSolution::getRecJetB() const 	  { return this->getJetB().getRecJet();}
TopJetType TtDilepEvtSolution::getRecJetBbar() const 	  { return this->getJetBbar().getRecJet();}
TopMET   TtDilepEvtSolution::getRecMET() const 	  { return this->getMET();}

reco::Particle TtDilepEvtSolution::getRecLepp() const {
  reco::Particle p;
  if (this->getWpDecay() == "muon") p = reco::Particle(0, this->getMuonLepp().p4(),math::XYZPoint());
  if (this->getWpDecay() == "electron") p = reco::Particle(0, this->getElectronLepp().p4(),math::XYZPoint());
  return p;
}

reco::Particle TtDilepEvtSolution::getRecLepm() const {
  reco::Particle p;
  if (this->getWmDecay() == "muon") p = reco::Particle(0, this->getMuonLepm().p4(),math::XYZPoint());
  if (this->getWmDecay() == "electron") p = reco::Particle(0, this->getElectronLepm().p4(),math::XYZPoint());
  return p;  
}

// return functions for calibrated fourvectors
TopJet TtDilepEvtSolution::getCalJetB() const 	 { return this->getJetB(); }
TopJet TtDilepEvtSolution::getCalJetBbar() const  { return this->getJetBbar(); }


   
