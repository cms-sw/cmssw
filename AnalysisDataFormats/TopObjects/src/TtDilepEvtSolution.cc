
// system include files

// user include files
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

void TtDilepEvtSolution::setGenEvt(std::vector<reco:: Candidate *> particles){

  if ((*particles[0]).pdgId() > 0) {
    genLepm = (reco::Particle) (*(particles[0]));
    genLepp = (reco::Particle) (*(particles[4]));
  }
  else {
    genLepp = (reco::Particle) (*(particles[0]));
    genLepm = (reco::Particle) (*(particles[4]));
  }
  
  if ((*particles[1]).pdgId() > 0) {
    genN = (reco::Particle) (*(particles[1]));
    genNbar = (reco::Particle) (*(particles[5]));
  }
  else {
    genNbar = (reco::Particle) (*(particles[1]));
    genN = (reco::Particle) (*(particles[5]));
  }
  
  if ((*particles[2]).pdgId() > 0) {
    genB = (reco::Particle) (*(particles[2]));
    genBbar = (reco::Particle) (*(particles[3]));
  }
  else {
    genBbar = (reco::Particle) (*(particles[2]));
    genB = (reco::Particle) (*(particles[3]));
  }
  
  if ((*particles[8]).pdgId() > 0) {
    genT = (reco::Particle) (*(particles[8]));
    genTbar = (reco::Particle) (*(particles[9]));
  }
  else {
    genTbar = (reco::Particle) (*(particles[8]));
    genT = (reco::Particle) (*(particles[9]));
  }
  
  if ((*particles[6]).charge() > 0) {
    genWp = (reco::Particle) (*(particles[6]));
    genWm = (reco::Particle) (*(particles[7]));
  }
  else {
    genWm = (reco::Particle) (*(particles[6]));
    genWp = (reco::Particle) (*(particles[7]));
  }
  
}

void TtDilepEvtSolution::setBestSol(bool bs)			{ bestSol     = bs;  }


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

JetType TtDilepEvtSolution::getRecJetB() const 	  { return this->getJetB().getRecJet();}
JetType TtDilepEvtSolution::getRecJetBbar() const 	  { return this->getJetBbar().getRecJet();}
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


   
