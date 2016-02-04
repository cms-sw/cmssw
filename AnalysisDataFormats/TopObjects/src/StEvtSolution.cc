//
// $Id: StEvtSolution.cc,v 1.11 2008/11/14 19:20:51 rwolf Exp $
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

StEvtSolution::StEvtSolution()
{
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
  changeBL_       = -999 ;
  bestSol_        = false;
}

StEvtSolution::~StEvtSolution()
{
}

//-------------------------------------------
// get calibrated base objects 
//-------------------------------------------
pat::Jet StEvtSolution::getBottom() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return bottom_->correctedJet("HAD", "B");
  } 
  else if(jetCorrScheme_==2){
    return bottom_->correctedJet("HAD", "B");
  }
  else{
    return *bottom_;
  }
}

pat::Jet StEvtSolution::getLight() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return light_->correctedJet("HAD", "UDS");
  }
  else if(jetCorrScheme_==2){
    return light_->correctedJet("HAD", "UDS");
  }
  else{
    return *light_;
  }
}

reco::Particle StEvtSolution::getLepW() const 
{
  // FIXME: the charge from the genevent
  reco::Particle p;
  if(this->getDecay() == "muon") p=reco::Particle(0, this->getMuon().p4()+this->getNeutrino().p4(), math::XYZPoint());
  if(this->getDecay() == "electron") p=reco::Particle(0, this->getElectron().p4()+this->getNeutrino().p4(), math::XYZPoint());
  return p;
}

reco::Particle StEvtSolution::getLept() const 
{
  // FIXME: the charge from the genevent
  reco::Particle p;
  if(this->getDecay() == "muon") p=reco::Particle(0, this->getMuon().p4()+this->getNeutrino().p4()+this->getBottom().p4(), math::XYZPoint());
  if(this->getDecay() == "electron") p=reco::Particle(0, this->getElectron().p4()+this->getNeutrino().p4()+this->getBottom().p4(), math::XYZPoint());
  return p;
}

//-------------------------------------------
// get the matched gen particles
//-------------------------------------------
// FIXME: provide defaults if the genevent is invalid
const reco::GenParticle * StEvtSolution::getGenBottom() const 
{ 
  if(!theGenEvt_) return 0; 
  else return theGenEvt_->decayB();
}

// FIXME: not implemented yet
// const reco::GenParticle * StEvtSolution::getGenLight() const 
// { 
//   if(!theGenEvt_) return 0; 
//   else return theGenEvt_->recoilQuark(); 
// }

const reco::GenParticle * StEvtSolution::getGenLepton() const 
{ 
  if(!theGenEvt_) return 0; 
  else return theGenEvt_->singleLepton(); 
}

const reco::GenParticle * StEvtSolution::getGenNeutrino() const 
{ 
  if(!theGenEvt_) return 0; 
  else return theGenEvt_->singleNeutrino(); 
}

const reco::GenParticle * StEvtSolution::getGenLepW() const 
{ 
  if (!theGenEvt_) return 0; 
  else return theGenEvt_->singleW(); 
}

const reco::GenParticle * StEvtSolution::getGenLept() const 
{ 
  if (!theGenEvt_) return 0; 
  else return theGenEvt_->singleTop(); 
}

//-------------------------------------------
// get uncalibrated reco objects
//-------------------------------------------
reco::Particle StEvtSolution::getRecLept() const 
{
  // FIXME: the charge from the genevent
  reco::Particle p;
  if(this->getDecay() == "muon") p=reco::Particle(0, this->getMuon().p4()+this->getNeutrino().p4()+this->getRecBottom().p4(), math::XYZPoint());
  if(this->getDecay() == "electron") p=reco::Particle(0, this->getElectron().p4()+this->getNeutrino().p4()+this->getRecBottom().p4(), math::XYZPoint());
  return p;
}

//-------------------------------------------
// get objects from kinematic fit
//-------------------------------------------
reco::Particle StEvtSolution::getFitLepW() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepton().p4()+this->getFitNeutrino().p4());
}

reco::Particle StEvtSolution::getFitLept() const 
{ 
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepton().p4()+this->getFitNeutrino().p4()+this->getFitBottom().p4());
}

//-------------------------------------------  
// set the generated event
//-------------------------------------------
void StEvtSolution::setGenEvt(const edm::Handle<StGenEvent> & aGenEvt){
  theGenEvt_ = edm::RefProd<StGenEvent>(aGenEvt);
}

//-------------------------------------------
// set other info on the event
//-------------------------------------------
void StEvtSolution::setScanValues(const std::vector<double> & val) {
  for(unsigned int i=0; i<val.size(); i++) scanValues_.push_back(val[i]);
}
