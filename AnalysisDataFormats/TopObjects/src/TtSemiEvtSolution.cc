//
// $Id: TtSemiEvtSolution.cc,v 1.22 2008/01/25 13:34:29 vadler Exp $
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

TtSemiEvtSolution::TtSemiEvtSolution() 
{
  jetCorrScheme_     = 0;
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

TtSemiEvtSolution::~TtSemiEvtSolution() 
{
}

//-------------------------------------------
// get calibrated base objects 
//------------------------------------------- 
pat::Jet TtSemiEvtSolution::getHadb() const 
{
  if (jetCorrScheme_ == 1) return hadb_->mcFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadb_->bCorrJet();
  else return *hadb_;
}

pat::Jet TtSemiEvtSolution::getHadp() const 
{
  if (jetCorrScheme_ == 1) return hadp_->mcFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadp_->wCorrJet();
  else return *hadp_;
}

pat::Jet TtSemiEvtSolution::getHadq() const 
{
  if (jetCorrScheme_ == 1) return hadq_->mcFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadq_->wCorrJet();
  else return *hadq_;
}

pat::Jet TtSemiEvtSolution::getLepb() const 
{
  if (jetCorrScheme_ == 1) return lepb_->mcFlavCorrJet(); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return lepb_->bCorrJet();
  else return *lepb_;
}

//-------------------------------------------
// get (un-)/calibrated reco objects
//-------------------------------------------
reco::Particle TtSemiEvtSolution::getRecHadt() const 
{
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4() + this->getRecHadb().p4());
}

reco::Particle TtSemiEvtSolution::getRecHadW() const 
{
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4());
}

reco::Particle TtSemiEvtSolution::getRecLept() const 
{
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getRecLepb().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getRecLepb().p4());
  return p;
}

reco::Particle TtSemiEvtSolution::getRecLepW() const 
{ 
  // FIXME: the charge from the genevent
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4());
  return p;
}

// FIXME: Why these functions??? Not needed!
  // methods to get calibrated objects 
reco::Particle TtSemiEvtSolution::getCalHadt() const 
{ 
  return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4() + this->getCalHadb().p4()); 
}

reco::Particle TtSemiEvtSolution::getCalHadW() const 
{ 
  return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4()); 
}

reco::Particle TtSemiEvtSolution::getCalLept() const 
{
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getCalLepb().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getCalLepb().p4());
  return p;
}

reco::Particle TtSemiEvtSolution::getCalLepW() const 
{
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4());
  return p;
}

//-------------------------------------------
// get objects from kinematic fit
//-------------------------------------------  
reco::Particle TtSemiEvtSolution::getFitHadt() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4() + this->getFitHadq().p4() + this->getFitHadb().p4());
}

reco::Particle TtSemiEvtSolution::getFitHadW() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4() + this->getFitHadq().p4());
}

reco::Particle TtSemiEvtSolution::getFitLept() const 
{ 
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepl().p4() + this->getFitLepn().p4() + this->getFitLepb().p4());
}

reco::Particle TtSemiEvtSolution::getFitLepW() const 
{ 
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitLepl().p4() + this->getFitLepn().p4());
}

//-------------------------------------------
// get info on the outcome of the signal 
// selection LR
//-------------------------------------------
double TtSemiEvtSolution::getLRSignalEvtObsVal(unsigned int selObs) const {
  double val = -999.;
  for(size_t o=0; o<lrSignalEvtVarVal_.size(); o++){
    if(lrSignalEvtVarVal_[o].first == selObs) val = lrSignalEvtVarVal_[o].second;
  }
  return val;
}

//-------------------------------------------
// get info on the outcome of the different 
// jet combination methods
//-------------------------------------------
double TtSemiEvtSolution::getLRJetCombObsVal(unsigned int selObs) const 
{
  double val = -999.;
  for(size_t o=0; o<lrJetCombVarVal_.size(); o++){
    if(lrJetCombVarVal_[o].first == selObs) val = lrJetCombVarVal_[o].second;
  }
  return val;
}

//-------------------------------------------  
// set the generated event
//-------------------------------------------
void TtSemiEvtSolution::setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt)
{
  if( !aGenEvt->isSemiLeptonic() ){
    edm::LogWarning( "TtGenEventNotFilled" ) << "genEvt is not semi-leptonic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}

//-------------------------------------------  
// set the outcome of the different jet 
// combination methods
//-------------------------------------------  
void TtSemiEvtSolution::setLRJetCombObservables(std::vector<std::pair<unsigned int, double> > varval) 
{
  lrJetCombVarVal_.clear();
  for(size_t ijc = 0; ijc<varval.size(); ijc++) lrJetCombVarVal_.push_back(varval[ijc]);
}

//-------------------------------------------  
// set the outcome of the signal selection LR
//-------------------------------------------  
void TtSemiEvtSolution::setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval) 
{
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}
