//
// $Id: TtHadEvtSolution.cc,v 1.5 2007/11/24 11:03:16 lowette Exp $
// adapted TtSemiEvtSolution.cc,v 1.13 2007/07/05 23:43:08 lowette Exp 
// for fully hadronic channel

#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// constructor
TtHadEvtSolution::TtHadEvtSolution() {
  jetCorrScheme_     = 0;
  sumAnglejp_        = -999.;
  angleHadp_         = -999.;
  angleHadq_         = -999.;
  angleHadb_         = -999.;
  angleHadj_         = -999.;
  angleHadk_         = -999.;
  angleHadbbar_      = -999.;
  changeW1Q_         = -999;
  changeW2Q_         = -999;
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
TtHadEvtSolution::~TtHadEvtSolution() {
}


// members to get original TopObjects 
TopJet TtHadEvtSolution::getHadb() const {
  if (jetCorrScheme_ == 1) return hadb_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadb_->getBCorrJet();
  else return *hadb_;
}
TopJet TtHadEvtSolution::getHadp() const {
  if (jetCorrScheme_ == 1) return hadp_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadp_->getWCorrJet();
  else return *hadp_;
}
TopJet TtHadEvtSolution::getHadq() const {
  if (jetCorrScheme_ == 1) return hadq_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadq_->getWCorrJet();
  else return *hadq_;
}
TopJet TtHadEvtSolution::getHadbbar() const {
  if (jetCorrScheme_ == 1) return hadbbar_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadbbar_->getBCorrJet();
  else return *hadbbar_;
}
TopJet TtHadEvtSolution::getHadj() const {
  if (jetCorrScheme_ == 1) return hadj_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadj_->getWCorrJet();
  else return *hadj_;
}
TopJet TtHadEvtSolution::getHadk() const {
  if (jetCorrScheme_ == 1) return hadk_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadk_->getWCorrJet();
  else return *hadk_;
}

// methods to get the MC matched particles
const edm::RefProd<TtGenEvent> & TtHadEvtSolution::getGenEvent() const { return theGenEvt_; }
const reco::GenParticle * TtHadEvtSolution::getGenHadb() const { if (!theGenEvt_) return 0; else return theGenEvt_->b(); }
const reco::GenParticle * TtHadEvtSolution::getGenHadbbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->bBar(); }
const reco::GenParticle * TtHadEvtSolution::getGenHadp() const { if (!theGenEvt_) return 0; else return theGenEvt_->quarkFromTop(); }
const reco::GenParticle * TtHadEvtSolution::getGenHadq() const { if (!theGenEvt_) return 0; else return theGenEvt_->quarkFromTopBar(); }
const reco::GenParticle * TtHadEvtSolution::getGenHadj() const { if (!theGenEvt_) return 0; else return theGenEvt_->quarkFromAntiTop(); }
const reco::GenParticle * TtHadEvtSolution::getGenHadk() const { if (!theGenEvt_) return 0; else return theGenEvt_->quarkFromAntiTopBar(); }

// return functions for non-calibrated fourvectors
// By definition pq and b are the top quark, jk and bbar the anti-top - check it makes sense ....
reco::Particle TtHadEvtSolution::getRecHadt() const {
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4() + this->getRecHadb().p4());
}
reco::Particle TtHadEvtSolution::getRecHadtbar() const {
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadj().p4() + this->getRecHadk().p4() + this->getRecHadbbar().p4());
}
reco::Particle TtHadEvtSolution::getRecHadW_plus() const {
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4());
}
reco::Particle TtHadEvtSolution::getRecHadW_minus() const {
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadj().p4() + this->getRecHadk().p4());
}

TopJetType     TtHadEvtSolution::getRecHadb() const { return this->getHadb().getRecJet(); }
TopJetType     TtHadEvtSolution::getRecHadbbar() const { return this->getHadbbar().getRecJet(); }
TopJetType     TtHadEvtSolution::getRecHadp() const { return this->getHadp().getRecJet(); }
TopJetType     TtHadEvtSolution::getRecHadq() const { return this->getHadq().getRecJet(); }
TopJetType     TtHadEvtSolution::getRecHadj() const { return this->getHadj().getRecJet(); }
TopJetType     TtHadEvtSolution::getRecHadk() const { return this->getHadk().getRecJet(); }

// return functions for calibrated fourvectors
reco::Particle TtHadEvtSolution::getCalHadt() const { return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4() + this->getCalHadb().p4()); }
reco::Particle TtHadEvtSolution::getCalHadtbar() const { return reco::Particle(0,this->getCalHadj().p4() + this->getCalHadk().p4() + this->getCalHadbbar().p4()); }
reco::Particle TtHadEvtSolution::getCalHadW_plus() const { return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4()); }
reco::Particle TtHadEvtSolution::getCalHadW_minus() const { return reco::Particle(0,this->getCalHadj().p4() + this->getCalHadk().p4()); }
TopJet         TtHadEvtSolution::getCalHadb() const { return this->getHadb(); }
TopJet         TtHadEvtSolution::getCalHadbbar() const { return this->getHadbbar(); }
TopJet         TtHadEvtSolution::getCalHadp() const { return this->getHadp(); }
TopJet         TtHadEvtSolution::getCalHadq() const { return this->getHadq(); }
TopJet         TtHadEvtSolution::getCalHadj() const { return this->getHadj(); }
TopJet         TtHadEvtSolution::getCalHadk() const { return this->getHadk(); }

// return functions for fitted fourvectors
reco::Particle TtHadEvtSolution::getFitHadt() const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4() + this->getFitHadq().p4() + this->getFitHadb().p4());
}
reco::Particle TtHadEvtSolution::getFitHadtbar() const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadj().p4() + this->getFitHadk().p4() + this->getFitHadbbar().p4());
}
reco::Particle TtHadEvtSolution::getFitHadW_plus() const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4() + this->getFitHadq().p4());
}
reco::Particle TtHadEvtSolution::getFitHadW_minus() const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadj().p4() + this->getFitHadk().p4());
}
TopParticle    TtHadEvtSolution::getFitHadb() const { return (fitHadb_.size()>0 ? fitHadb_.front() : TopParticle()); }
TopParticle    TtHadEvtSolution::getFitHadbbar() const { return (fitHadbbar_.size()>0 ? fitHadbbar_.front() : TopParticle()); }
TopParticle    TtHadEvtSolution::getFitHadp() const { return (fitHadp_.size()>0 ? fitHadp_.front() : TopParticle()); }
TopParticle    TtHadEvtSolution::getFitHadq() const { return (fitHadq_.size()>0 ? fitHadq_.front() : TopParticle()); }
TopParticle    TtHadEvtSolution::getFitHadj() const { return (fitHadj_.size()>0 ? fitHadj_.front() : TopParticle()); }
TopParticle    TtHadEvtSolution::getFitHadk() const { return (fitHadk_.size()>0 ? fitHadk_.front() : TopParticle()); }

// method to get info on the outcome of the signal selection LR
double TtHadEvtSolution::getLRJetCombObsVal(unsigned int selObs) const {
  double val = -999.;
  for(size_t o=0; o<lrJetCombVarVal_.size(); o++){
    if(lrJetCombVarVal_[o].first == selObs) val = lrJetCombVarVal_[o].second;
  }
  return val;
}
// method to get info on the outcome of the different jet combination methods
double TtHadEvtSolution::getLRSignalEvtObsVal(unsigned int selObs) const {
  double val = -999.;
  for(size_t o=0; o<lrSignalEvtVarVal_.size(); o++){
    if(lrSignalEvtVarVal_[o].first == selObs) val = lrSignalEvtVarVal_[o].second;
  }
  return val;
}


// method to set the generated event
void TtHadEvtSolution::setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt){
  if( !aGenEvt->isFullHadronic() ){ 
    edm::LogWarning( "TtGenEventNotFilled" ) << "genEvt is not fully hadronic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}


// methods to set the basic TopObjects
void TtHadEvtSolution::setJetCorrectionScheme(int jetCorrScheme) {
  jetCorrScheme_ = jetCorrScheme;
}
void TtHadEvtSolution::setHadb(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadb_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtHadEvtSolution::setHadbbar(const edm::Handle<std::vector<TopJet> > & jh, int i)       { hadbbar_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtHadEvtSolution::setHadp(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadp_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtHadEvtSolution::setHadq(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadq_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtHadEvtSolution::setHadj(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadj_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtHadEvtSolution::setHadk(const edm::Handle<std::vector<TopJet> > & jh, int i)          { hadk_ = edm::Ref<std::vector<TopJet> >(jh, i); }

// methods to set the fitted particles
void TtHadEvtSolution::setFitHadb(const TopParticle & aFitHadb) { fitHadb_.clear(); fitHadb_.push_back(aFitHadb); }
void TtHadEvtSolution::setFitHadbbar(const TopParticle & aFitHadbbar) { fitHadbbar_.clear(); fitHadbbar_.push_back(aFitHadbbar); }
void TtHadEvtSolution::setFitHadp(const TopParticle & aFitHadp) { fitHadp_.clear(); fitHadp_.push_back(aFitHadp); }
void TtHadEvtSolution::setFitHadq(const TopParticle & aFitHadq) { fitHadq_.clear(); fitHadq_.push_back(aFitHadq); }
void TtHadEvtSolution::setFitHadj(const TopParticle & aFitHadj) { fitHadj_.clear(); fitHadj_.push_back(aFitHadj); }
void TtHadEvtSolution::setFitHadk(const TopParticle & aFitHadk) { fitHadk_.clear(); fitHadk_.push_back(aFitHadk); }

// methods to set the info on the matching
void TtHadEvtSolution::setMCBestSumAngles(double sdr) { sumAnglejp_ = sdr; }
void TtHadEvtSolution::setMCBestAngleHadp(double adr) { angleHadp_ = adr; }
void TtHadEvtSolution::setMCBestAngleHadq(double adr) { angleHadq_ = adr; }
void TtHadEvtSolution::setMCBestAngleHadb(double adr) { angleHadb_ = adr; }
void TtHadEvtSolution::setMCBestAngleHadj(double adr) { angleHadj_ = adr; }
void TtHadEvtSolution::setMCBestAngleHadk(double adr) { angleHadk_ = adr; }
void TtHadEvtSolution::setMCBestAngleHadbbar(double adr) { angleHadbbar_ = adr; }

void TtHadEvtSolution::setMCChangeW1Q(int w1q)          { changeW1Q_ = w1q; }
void TtHadEvtSolution::setMCChangeW2Q(int w2q)          { changeW2Q_ = w2q; }  
// methods to set the kinfit parametrisations of each type of object
void TtHadEvtSolution::setJetParametrisation(int jp)    { jetParam_ = jp; }


// method to set the prob. of the chi2 value resulting from the kinematic fit 
void TtHadEvtSolution::setProbChi2(double c) { probChi2_ = c; }

// methods to set the outcome of the different jet combination methods
void TtHadEvtSolution::setMCBestJetComb(int mcbs)    { mcBestJetComb_ = mcbs; }
void TtHadEvtSolution::setSimpleBestJetComb(int sbs) { simpleBestJetComb_ = sbs;  }
void TtHadEvtSolution::setLRBestJetComb(int lrbs)    { lrBestJetComb_ = lrbs;  }
void TtHadEvtSolution::setLRJetCombObservables(std::vector<std::pair<unsigned int, double> > varval) {
  lrJetCombVarVal_.clear();
  for(size_t ijc = 0; ijc<varval.size(); ijc++) lrJetCombVarVal_.push_back(varval[ijc]);
}
void TtHadEvtSolution::setLRJetCombLRval(double clr) {lrJetCombLRval_ = clr;}
void TtHadEvtSolution::setLRJetCombProb(double plr)  {lrJetCombProb_ = plr;}


// methods to set the outcome of the signal selection LR
void TtHadEvtSolution::setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval) {
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}
void TtHadEvtSolution::setLRSignalEvtLRval(double clr) {lrSignalEvtLRval_ = clr;}
void TtHadEvtSolution::setLRSignalEvtProb(double plr)  {lrSignalEvtProb_ = plr;}
  

