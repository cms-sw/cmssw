//
// $Id: TtSemiEvtSolution.cc,v 1.16 2007/07/26 08:42:46 lowette Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// constructor
TtSemiEvtSolution::TtSemiEvtSolution() {
  sumAnglejp_        = -999.;
  angleHadp_         = -999.;
  angleHadq_         = -999.;
  angleHadb_         = -999.;
  angleLepb_         = -999.;
  changeWQ_          = -999;
  probChi2_          = -999.;
  mcBestJetComb_     = -999;
  simpleBestJetComb_ = -999;
  lrBestJetComb_     = -999;
  lrJetCombLRval_    = -999.;
  lrJetCombProb_     = -999.;
  lrSignalEvtLRval_  = -999.;
  lrSignalEvtProb_   = -999.;
}


/// destructor
TtSemiEvtSolution::~TtSemiEvtSolution() {
}


// members to get original TopObjects 
TopJet      TtSemiEvtSolution::getHadb() const     { return *hadb_; }
TopJet      TtSemiEvtSolution::getHadp() const     { return *hadp_; }
TopJet      TtSemiEvtSolution::getHadq() const     { return *hadq_; }
TopJet      TtSemiEvtSolution::getLepb() const     { return *lepb_; }
TopMuon     TtSemiEvtSolution::getMuon() const     { return *muon_; }
TopElectron TtSemiEvtSolution::getElectron() const { return *electron_; }
TopMET      TtSemiEvtSolution::getMET() const      { return *met_; }


// methods to get the MC matched particles
// FIXME: provide defaults if the genevent is invalid
const reco::Candidate * TtSemiEvtSolution::getGenHadt() const { return theGenEvt_->hadronicDecayTop(); }
const reco::Candidate * TtSemiEvtSolution::getGenHadW() const { return theGenEvt_->hadronicDecayW(); }
const reco::Candidate * TtSemiEvtSolution::getGenHadb() const { return theGenEvt_->hadronicDecayB(); }
const reco::Candidate * TtSemiEvtSolution::getGenHadp() const { return theGenEvt_->hadronicDecayQuark(); }
const reco::Candidate * TtSemiEvtSolution::getGenHadq() const { return theGenEvt_->hadronicDecayQuarkBar(); }
const reco::Candidate * TtSemiEvtSolution::getGenLept() const { return theGenEvt_->leptonicDecayTop(); }
const reco::Candidate * TtSemiEvtSolution::getGenLepW() const { return theGenEvt_->leptonicDecayW(); }
const reco::Candidate * TtSemiEvtSolution::getGenLepb() const { return theGenEvt_->leptonicDecayB(); }
const reco::Candidate * TtSemiEvtSolution::getGenLepl() const { return theGenEvt_->singleLepton(); }
const reco::Candidate * TtSemiEvtSolution::getGenLepn() const { return theGenEvt_->singleNeutrino(); }


// return functions for non-calibrated fourvectors
reco::Particle TtSemiEvtSolution::getRecHadt() const {
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4() + this->getRecHadb().p4());
}
reco::Particle TtSemiEvtSolution::getRecHadW() const {
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4());
}
TopJetType     TtSemiEvtSolution::getRecHadb() const { return this->getHadb().getRecJet(); }
TopJetType     TtSemiEvtSolution::getRecHadp() const { return this->getHadp().getRecJet(); }
TopJetType     TtSemiEvtSolution::getRecHadq() const { return this->getHadq().getRecJet(); }
reco::Particle TtSemiEvtSolution::getRecLept() const { 
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getRecLepb().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getRecLepb().p4());
  return p;
}
reco::Particle TtSemiEvtSolution::getRecLepW() const { 
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4());
  return p;
}
TopJetType     TtSemiEvtSolution::getRecLepb() const { return this->getLepb().getRecJet(); }
TopMuon        TtSemiEvtSolution::getRecLepm() const { return this->getMuon(); }
TopElectron    TtSemiEvtSolution::getRecLepe() const { return this->getElectron(); }
TopMET         TtSemiEvtSolution::getRecLepn() const { return this->getMET();  }


// FIXME: Why these functions??? Not needed!
// return functions for calibrated fourvectors
reco::Particle TtSemiEvtSolution::getCalHadt() const { return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4() + this->getCalHadb().p4()); }
reco::Particle TtSemiEvtSolution::getCalHadW() const { return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4()); }
TopJet         TtSemiEvtSolution::getCalHadb() const { return this->getHadb(); }
TopJet         TtSemiEvtSolution::getCalHadp() const { return this->getHadp(); }
TopJet         TtSemiEvtSolution::getCalHadq() const { return this->getHadq(); }
reco::Particle TtSemiEvtSolution::getCalLept() const {
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getCalLepb().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getCalLepb().p4());
  return p;
}
reco::Particle TtSemiEvtSolution::getCalLepW() const {
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4());
  return p;
}
TopJet         TtSemiEvtSolution::getCalLepb() const { return this->getLepb(); }
TopMuon        TtSemiEvtSolution::getCalLepm() const { return this->getMuon(); }
TopElectron    TtSemiEvtSolution::getCalLepe() const { return this->getElectron(); }
TopMET         TtSemiEvtSolution::getCalLepn() const { return this->getMET();  }


// return functions for fitted fourvectors
reco::Particle TtSemiEvtSolution::getFitHadt() const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4() + this->getFitHadq().p4() + this->getFitHadb().p4());
}
reco::Particle TtSemiEvtSolution::getFitHadW() const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4() + this->getFitHadq().p4());
}
TopParticle    TtSemiEvtSolution::getFitHadb() const { return (fitHadb_.size()>0 ? fitHadb_.front() : TopParticle()); }
TopParticle    TtSemiEvtSolution::getFitHadp() const { return (fitHadp_.size()>0 ? fitHadp_.front() : TopParticle()); }
TopParticle    TtSemiEvtSolution::getFitHadq() const { return (fitHadq_.size()>0 ? fitHadq_.front() : TopParticle()); }
reco::Particle TtSemiEvtSolution::getFitLept() const { 
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepl().p4() + this->getFitLepn().p4() + this->getFitLepb().p4());
}
reco::Particle TtSemiEvtSolution::getFitLepW() const { 
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepl().p4() + this->getFitLepn().p4());
}
TopParticle    TtSemiEvtSolution::getFitLepb() const { return (fitLepb_.size()>0 ? fitLepb_.front() : TopParticle()); }
TopParticle    TtSemiEvtSolution::getFitLepl() const { return (fitLepl_.size()>0 ? fitLepl_.front() : TopParticle()); }
TopParticle    TtSemiEvtSolution::getFitLepn() const { return (fitLepn_.size()>0 ? fitLepn_.front() : TopParticle()); }
   

// method to get info on the outcome of the signal selection LR
double TtSemiEvtSolution::getLRJetCombObsVal(unsigned int selObs) const {
  double val = -999.;
  for(size_t o=0; o<lrJetCombVarVal_.size(); o++){
    if(lrJetCombVarVal_[o].first == selObs) val = lrJetCombVarVal_[o].second;
  }
  return val;
}
// method to get info on the outcome of the different jet combination methods
double TtSemiEvtSolution::getLRSignalEvtObsVal(unsigned int selObs) const {
  double val = -999.;
  for(size_t o=0; o<lrSignalEvtVarVal_.size(); o++){
    if(lrSignalEvtVarVal_[o].first == selObs) val = lrSignalEvtVarVal_[o].second;
  }
  return val;
}


// method to set the generated event
void TtSemiEvtSolution::setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt){
  if( !aGenEvt->isSemiLeptonic() ){
    edm::LogWarning( "TtGenEventNotFilled" ) << "genEvt is not semi-leptonic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}


// methods to set the basic TopObjects
void TtSemiEvtSolution::setHadb(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadb_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtSemiEvtSolution::setHadp(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadp_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtSemiEvtSolution::setHadq(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadq_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtSemiEvtSolution::setLepb(const edm::Handle<std::vector<TopJet> > & jh, int i)          { lepb_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtSemiEvtSolution::setMuon(const edm::Handle<std::vector<TopMuon> > & mh, int i)         { muon_ = edm::Ref<std::vector<TopMuon> >(mh, i); decay_ = "muon"; }
void TtSemiEvtSolution::setElectron(const edm::Handle<std::vector<TopElectron> > & eh, int i) { electron_ = edm::Ref<std::vector<TopElectron> >(eh, i); decay_ = "electron"; }
void TtSemiEvtSolution::setMET(const edm::Handle<std::vector<TopMET> > & nh, int i)           { met_ = edm::Ref<std::vector<TopMET> >(nh, i); }


// methods to set the fitted particles
void TtSemiEvtSolution::setFitHadb(const TopParticle & aFitHadb) { fitHadb_.clear(); fitHadb_.push_back(aFitHadb); }
void TtSemiEvtSolution::setFitHadp(const TopParticle & aFitHadp) { fitHadp_.clear(); fitHadp_.push_back(aFitHadp); }
void TtSemiEvtSolution::setFitHadq(const TopParticle & aFitHadq) { fitHadq_.clear(); fitHadq_.push_back(aFitHadq); }
void TtSemiEvtSolution::setFitLepb(const TopParticle & aFitLepb) { fitLepb_.clear(); fitLepb_.push_back(aFitLepb); }
void TtSemiEvtSolution::setFitLepl(const TopParticle & aFitLepl) { fitLepl_.clear(); fitLepl_.push_back(aFitLepl); }
void TtSemiEvtSolution::setFitLepn(const TopParticle & aFitLepn) { fitLepn_.clear(); fitLepn_.push_back(aFitLepn); }


// methods to set the info on the matching
void TtSemiEvtSolution::setMCBestSumAngles(double sdr) { sumAnglejp_ = sdr; }
void TtSemiEvtSolution::setMCBestAngleHadp(double adr) { angleHadp_ = adr; }
void TtSemiEvtSolution::setMCBestAngleHadq(double adr) { angleHadq_ = adr; }
void TtSemiEvtSolution::setMCBestAngleHadb(double adr) { angleHadb_ = adr; }
void TtSemiEvtSolution::setMCBestAngleLepb(double adr) { angleLepb_ = adr; }
void TtSemiEvtSolution::setMCChangeWQ(int wq)          { changeWQ_ = wq; }


// methods to set the kinfit parametrisations of each type of object
void TtSemiEvtSolution::setJetParametrisation(int jp)    { jetParam_ = jp; }
void TtSemiEvtSolution::setLeptonParametrisation(int lp) { lepParam_ = lp; }
void TtSemiEvtSolution::setMETParametrisation(int mp)    { metParam_ = mp; }


// method to set the prob. of the chi2 value resulting from the kinematic fit 
void TtSemiEvtSolution::setProbChi2(double c) { probChi2_ = c; }


// methods to set the outcome of the different jet combination methods
void TtSemiEvtSolution::setMCBestJetComb(int mcbs)    { mcBestJetComb_ = mcbs; }
void TtSemiEvtSolution::setSimpleBestJetComb(int sbs) { simpleBestJetComb_ = sbs;  }
void TtSemiEvtSolution::setLRBestJetComb(int lrbs)    { lrBestJetComb_ = lrbs;  }
void TtSemiEvtSolution::setLRJetCombObservables(std::vector<std::pair<unsigned int, double> > varval) {
  lrJetCombVarVal_.clear();
  for(size_t ijc = 0; ijc<varval.size(); ijc++) lrJetCombVarVal_.push_back(varval[ijc]);
}
void TtSemiEvtSolution::setLRJetCombLRval(double clr) {lrJetCombLRval_ = clr;}
void TtSemiEvtSolution::setLRJetCombProb(double plr)  {lrJetCombProb_ = plr;}


// methods to set the outcome of the signal selection LR
void TtSemiEvtSolution::setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval) {
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}
void TtSemiEvtSolution::setLRSignalEvtLRval(double clr) {lrSignalEvtLRval_ = clr;}
void TtSemiEvtSolution::setLRSignalEvtProb(double plr)  {lrSignalEvtProb_ = plr;}
  

