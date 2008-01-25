//
// $Id: TtDilepEvtSolution.cc,v 1.14 2008/01/17 10:08:03 speer Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// constructor
TtDilepEvtSolution::TtDilepEvtSolution() {
  jetCorrScheme_ = 0;
  wpDecay_ = "NotDefined";
  wmDecay_ = "NotDefined";
  bestSol_ = false;
  topmass_ = 0.;
  weightmax_ = 0.;
}


/// destructor
TtDilepEvtSolution::~TtDilepEvtSolution() {
}


// members to get original TopObjects 
TopJet      TtDilepEvtSolution::getJetB() const      {
  if (jetCorrScheme_ == 1) return jetB_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return jetB_->getBCorrJet();
  else return *jetB_;
}
TopJet      TtDilepEvtSolution::getJetBbar() const   {
  if (jetCorrScheme_ == 1) return jetBbar_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return jetBbar_->getBCorrJet();
  else return *jetBbar_;
}
TopElectron TtDilepEvtSolution::getElectronp() const { return *elecp_; }
TopElectron TtDilepEvtSolution::getElectronm() const { return *elecm_; }
TopMuon     TtDilepEvtSolution::getMuonp() const     { return *muonp_; }
TopMuon     TtDilepEvtSolution::getMuonm() const     { return *muonm_; }
TopTau      TtDilepEvtSolution::getTaup() const      { return *taup_; }
TopTau      TtDilepEvtSolution::getTaum() const      { return *taum_; }
TopMET      TtDilepEvtSolution::getMET() const       { return *met_; }


// methods to get the MC matched particles
const edm::RefProd<TtGenEvent> & TtDilepEvtSolution::getGenEvent() const { return theGenEvt_; }
const reco::GenParticle * TtDilepEvtSolution::getGenT() const    { if (!theGenEvt_) return 0; else return theGenEvt_->top(); }
const reco::GenParticle * TtDilepEvtSolution::getGenWp() const   { if (!theGenEvt_) return 0; else return theGenEvt_->wPlus(); }
const reco::GenParticle * TtDilepEvtSolution::getGenB() const    { if (!theGenEvt_) return 0; else return theGenEvt_->b(); }
const reco::GenParticle * TtDilepEvtSolution::getGenLepp() const { if (!theGenEvt_) return 0; else return theGenEvt_->leptonBar(); }
const reco::GenParticle * TtDilepEvtSolution::getGenN() const    { if (!theGenEvt_) return 0; else return theGenEvt_->neutrino(); }
const reco::GenParticle * TtDilepEvtSolution::getGenTbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->topBar(); }
const reco::GenParticle * TtDilepEvtSolution::getGenWm() const   { if (!theGenEvt_) return 0; else return theGenEvt_->wMinus(); }
const reco::GenParticle * TtDilepEvtSolution::getGenBbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->bBar(); }
const reco::GenParticle * TtDilepEvtSolution::getGenLepm() const { if (!theGenEvt_) return 0; else return theGenEvt_->lepton(); }
const reco::GenParticle * TtDilepEvtSolution::getGenNbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->neutrinoBar(); }

// methods to explicitly get reconstructed and calibrated objects 
TopJetType TtDilepEvtSolution::getRecJetB() const    { return this->getJetB().getRecJet(); }
TopJet     TtDilepEvtSolution::getCalJetB() const    { return this->getJetB(); }
TopJetType TtDilepEvtSolution::getRecJetBbar() const { return this->getJetBbar().getRecJet(); }
TopJet     TtDilepEvtSolution::getCalJetBbar() const { return this->getJetBbar(); }


// method to set the generated event
void TtDilepEvtSolution::setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt) {
  if( !aGenEvt->isFullLeptonic() ) {
    edm::LogInfo( "TtGenEventNotFilled" ) << "genEvt is not di-leptonic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}


// methods to set the basic TopObjects
void TtDilepEvtSolution::setJetCorrectionScheme(int jetCorrScheme) {
  jetCorrScheme_ = jetCorrScheme;
}
void TtDilepEvtSolution::setB(const edm::Handle<std::vector<TopJet> > & jh, int i)              { jetB_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtDilepEvtSolution::setBbar(const edm::Handle<std::vector<TopJet> > & jh, int i)           { jetBbar_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtDilepEvtSolution::setTaup(const edm::Handle<std::vector<TopTau> > & mh, int i)         { taup_ = edm::Ref<std::vector<TopTau> >(mh, i); wpDecay_ = "tau"; }
void TtDilepEvtSolution::setTaum(const edm::Handle<std::vector<TopTau> > & mh, int i)         { taum_ = edm::Ref<std::vector<TopTau> >(mh, i); wmDecay_ = "tau"; }
void TtDilepEvtSolution::setMuonp(const edm::Handle<std::vector<TopMuon> > & mh, int i)         { muonp_ = edm::Ref<std::vector<TopMuon> >(mh, i); wpDecay_ = "muon"; }
void TtDilepEvtSolution::setMuonm(const edm::Handle<std::vector<TopMuon> > & mh, int i)         { muonm_ = edm::Ref<std::vector<TopMuon> >(mh, i); wmDecay_ = "muon"; }
void TtDilepEvtSolution::setElectronp(const edm::Handle<std::vector<TopElectron> > & eh, int i) { elecp_ = edm::Ref<std::vector<TopElectron> >(eh, i); wpDecay_ = "electron"; }
void TtDilepEvtSolution::setElectronm(const edm::Handle<std::vector<TopElectron> > & eh, int i) { elecm_ = edm::Ref<std::vector<TopElectron> >(eh, i); wmDecay_ = "electron"; }
void TtDilepEvtSolution::setMET(const edm::Handle<std::vector<TopMET> > & nh, int i)            { met_ = edm::Ref<std::vector<TopMET> >(nh, i); }

// the residual (for matched events)
double TtDilepEvtSolution::getResidual() const
{
  double distance = 0.;
  if(!getGenB() || !getGenBbar()) return distance;
  distance += DeltaR<reco::Particle,reco::GenParticle>()(getCalJetB(),*getGenB());
  distance += DeltaR<reco::Particle,reco::GenParticle>()(getCalJetBbar(),*getGenBbar());
  return distance;
}

// miscellaneous methods
void TtDilepEvtSolution::setBestSol(bool bs)       { bestSol_ = bs; }
void TtDilepEvtSolution::setRecTopMass(double j)   { topmass_ = j; }
void TtDilepEvtSolution::setRecWeightMax(double j) { weightmax_ = j; }

// method to get info on the outcome of the different jet combination methods
double TtDilepEvtSolution::getLRSignalEvtObsVal(unsigned int selObs) const {
  double val = -999.;
  for(size_t i=0; i<lrSignalEvtVarVal_.size(); i++){
    if(lrSignalEvtVarVal_[i].first == selObs) val = lrSignalEvtVarVal_[i].second;
  }
  return val;
}

// methods to set the outcome of the signal selection LR
void TtDilepEvtSolution::setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval) {
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}


void TtDilepEvtSolution::setLRSignalEvtLRval(double clr) {lrSignalEvtLRval_ = clr;}
void TtDilepEvtSolution::setLRSignalEvtProb(double plr)  {lrSignalEvtProb_ = plr;}

reco::Particle TtDilepEvtSolution::getLeptPos() const {
  reco::Particle p;
  if (wpDecay_ == "muon")     {
    p = reco::Particle(+1, getMuonp().p4() );
    p.setPdgId(-11);
  }
  if (wpDecay_ == "electron") {
    p = reco::Particle(+1, getElectronp().p4() );
    p.setPdgId(-13);
  }
  if (wmDecay_ == "tau") {
    p = reco::Particle(+1, getTaup().p4() );
    p.setPdgId(-15);
  }
  return p;
}

reco::Particle TtDilepEvtSolution::getLeptNeg() const {
  reco::Particle p;
  if (wmDecay_ == "electron") {
    p = reco::Particle(-1, getElectronm().p4() );
    p.setPdgId(11);
  }
  if (wmDecay_ == "muon")     {
    p = reco::Particle(-1, getMuonm().p4() );
    p.setPdgId(13);
  }
  if (wmDecay_ == "tau") {
    p = reco::Particle(-1, getTaum().p4() );
    p.setPdgId(15);
  }
  return p;
}
