//
// $Id: StEvtSolution.cc,v 1.8 2007/11/24 11:03:16 lowette Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// constructor
StEvtSolution::StEvtSolution() {
  jetCorrScheme_  = 0;
  chi2Prob_       = -999.;
  pTrueCombExist_ = -999.;
  pTrueBJetSel_   = -999.;
  pTrueBhadrSel_  = -999.;
  pTrueJetComb_   = -999.;
  signalPur_      = -999.;
  signalLRTot_    = -999.;
  sumDeltaRjp_    = -999.;
  deltaRB_        = -999.;
  deltaRL_        = -999.;
  changeBL_       = -999;
  bestSol_        = false;
}


/// destructor
StEvtSolution::~StEvtSolution() {
}



// members to get original TopObjects 
TopJet         StEvtSolution::getBottom()   const {
  if (jetCorrScheme_ == 1) return bottom_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return bottom_->getBCorrJet();
  else return *bottom_;
}
TopJet         StEvtSolution::getLight()    const {
  if (jetCorrScheme_ == 1) return light_->getMCFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return light_->getUdsCorrJet();
  else return *light_;
}
TopMuon        StEvtSolution::getMuon()     const { return *muon_; }
TopElectron    StEvtSolution::getElectron() const { return *electron_; }
TopMET         StEvtSolution::getNeutrino() const { return *neutrino_; }
reco::Particle StEvtSolution::getLepW()     const {
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0, this->getMuon().p4()     + this->getNeutrino().p4(), math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0, this->getElectron().p4() + this->getNeutrino().p4(), math::XYZPoint());
  return p;
}
reco::Particle StEvtSolution::getLept()     const {
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0, this->getMuon().p4()     + this->getNeutrino().p4() + this->getBottom().p4(), math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0, this->getElectron().p4() + this->getNeutrino().p4() + this->getBottom().p4(), math::XYZPoint());
  return p;
}


// methods to get the MC matched particles
// FIXME: provide defaults if the genevent is invalid
const edm::RefProd<StGenEvent> & StEvtSolution::getGenEvent() const { return theGenEvt_; }
const reco::GenParticle * StEvtSolution::getGenBottom()   const { if (!theGenEvt_) return 0; else return theGenEvt_->decayB(); }
// not implemented yet
//const reco::GenParticle * StEvtSolution::getGenLight()    const { if (!theGenEvt_) return 0; else return theGenEvt_->recoilQuark(); }
const reco::GenParticle * StEvtSolution::getGenLepton()   const { if (!theGenEvt_) return 0; else return theGenEvt_->singleLepton(); }
const reco::GenParticle * StEvtSolution::getGenNeutrino() const { if (!theGenEvt_) return 0; else return theGenEvt_->singleNeutrino(); }
const reco::GenParticle * StEvtSolution::getGenLepW()     const { if (!theGenEvt_) return 0; else return theGenEvt_->singleW(); }
const reco::GenParticle * StEvtSolution::getGenLept()     const { if (!theGenEvt_) return 0; else return theGenEvt_->singleTop(); }


// return functions for reconstructed fourvectors
TopJetType     StEvtSolution::getRecBottom()   const { return this->getBottom().getRecJet(); }
TopJetType     StEvtSolution::getRecLight()    const { return this->getLight().getRecJet(); }
TopMuon        StEvtSolution::getRecMuon()     const { return this->getMuon(); }
TopElectron    StEvtSolution::getRecElectron() const { return this->getElectron(); }
TopMET         StEvtSolution::getRecNeutrino() const { return this->getNeutrino(); }
reco::Particle StEvtSolution::getRecLepW()     const { return this->getLepW(); }
reco::Particle StEvtSolution::getRecLept()     const {
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0, this->getMuon().p4()     + this->getNeutrino().p4() + this->getRecBottom().p4(), math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0, this->getElectron().p4() + this->getNeutrino().p4() + this->getRecBottom().p4(), math::XYZPoint());
  return p;
}


// return functions for fitted fourvectors
TopParticle    StEvtSolution::getFitBottom()   const { return (fitBottom_.size()>0   ? fitBottom_.front()   : TopParticle()); }
TopParticle    StEvtSolution::getFitLight()    const { return (fitLight_.size()>0    ? fitLight_.front()    : TopParticle()); }
TopParticle    StEvtSolution::getFitLepton()   const { return (fitLepton_.size()>0   ? fitLepton_.front()   : TopParticle()); }
TopParticle    StEvtSolution::getFitNeutrino() const { return (fitNeutrino_.size()>0 ? fitNeutrino_.front() : TopParticle()); }
reco::Particle StEvtSolution::getFitLepW()     const {
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepton().p4() + this->getFitNeutrino().p4());
}
reco::Particle StEvtSolution::getFitLept()     const { 
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepton().p4() + this->getFitNeutrino().p4() + this->getFitBottom().p4());
}


// method to set the generated event
void StEvtSolution::setGenEvt(const edm::Handle<StGenEvent> & aGenEvt){
  theGenEvt_ = edm::RefProd<StGenEvent>(aGenEvt);
}


// methods to set the basic TopObjects
void StEvtSolution::setJetCorrectionScheme(int jetCorrScheme) {
  jetCorrScheme_ = jetCorrScheme;
}
void StEvtSolution::setBottom(const edm::Handle<std::vector<TopJet> > & jh, int i)        { bottom_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void StEvtSolution::setLight(const edm::Handle<std::vector<TopJet> > & jh, int i)         { light_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void StEvtSolution::setMuon(const edm::Handle<std::vector<TopMuon> > & mh, int i)         { muon_ = edm::Ref<std::vector<TopMuon> >(mh, i); decay_ = "muon"; }
void StEvtSolution::setElectron(const edm::Handle<std::vector<TopElectron> > & eh, int i) { electron_ = edm::Ref<std::vector<TopElectron> >(eh, i); decay_ = "electron"; }
void StEvtSolution::setNeutrino(const edm::Handle<std::vector<TopMET> > & nh, int i)      { neutrino_ = edm::Ref<std::vector<TopMET> >(nh, i); }


// methods to set the fitted particles
void StEvtSolution::setFitBottom(const TopParticle & aFitBottom)     { fitBottom_.clear();   fitBottom_.push_back(aFitBottom); }
void StEvtSolution::setFitLight(const TopParticle & aFitLight)       { fitLight_.clear();    fitLight_.push_back(aFitLight); }
void StEvtSolution::setFitLepton(const TopParticle & aFitLepton)     { fitLepton_.clear();   fitLepton_.push_back(aFitLepton); }
void StEvtSolution::setFitNeutrino(const TopParticle & aFitNeutrino) { fitNeutrino_.clear(); fitNeutrino_.push_back(aFitNeutrino); }


    // methods to set other info on the event
void StEvtSolution::setChi2Prob(double c)         { chi2Prob_ = c; }
void StEvtSolution::setScanValues(const std::vector<double> & v) {
  for(unsigned int i=0; i<v.size(); i++) scanValues_.push_back(v[i]);
}
void StEvtSolution::setPtrueCombExist(double pce) { pTrueCombExist_ = pce; }
void StEvtSolution::setPtrueBJetSel(double pbs)   { pTrueBJetSel_ = pbs; }
void StEvtSolution::setPtrueBhadrSel(double pbh)  { pTrueBhadrSel_ = pbh; }
void StEvtSolution::setPtrueJetComb(double pt)    { pTrueJetComb_ = pt; }
void StEvtSolution::setSignalPurity(double c)     { signalPur_ = c; }
void StEvtSolution::setSignalLRTot(double c)      { signalLRTot_ = c; }
void StEvtSolution::setSumDeltaRjp(double sdr)    { sumDeltaRjp_ = sdr; }
void StEvtSolution::setDeltaRB(double adr)        { deltaRB_ = adr; }
void StEvtSolution::setDeltaRL(double adr)        { deltaRL_ = adr; }
void StEvtSolution::setChangeBL(int bl)	          { changeBL_ = bl;  }
void StEvtSolution::setBestSol(bool bs)	          { bestSol_ = bs;  }

