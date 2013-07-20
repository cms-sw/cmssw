//
// $Id: TtHadEvtSolution.cc,v 1.9 2013/04/19 22:13:23 wmtan Exp $
// adapted TtSemiEvtSolution.cc,v 1.13 2007/07/05 23:43:08 lowette Exp 
// for fully hadronic channel

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

TtHadEvtSolution::TtHadEvtSolution() 
{
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

TtHadEvtSolution::~TtHadEvtSolution() 
{
}

//-------------------------------------------
// get calibrated base objects 
//-------------------------------------------
pat::Jet TtHadEvtSolution::getHadb() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return hadb_->correctedJet("HAD", "B");
  }
  else if(jetCorrScheme_==2){
    return hadb_->correctedJet("HAD", "B");
  }
  else{
    return *hadb_;
  }
}

pat::Jet TtHadEvtSolution::getHadp() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return hadp_->correctedJet("HAD", "UDS");
  }
  else if(jetCorrScheme_==2){
    return hadp_->correctedJet("HAD", "UDS");
  }
  else{
    return *hadp_;
  }
}

pat::Jet TtHadEvtSolution::getHadq() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return hadq_->correctedJet("HAD", "UDS");
  }
  else if(jetCorrScheme_==2){
    return hadq_->correctedJet("HAD", "UDS");
  }
  else{
    return *hadq_;
  }
}

pat::Jet TtHadEvtSolution::getHadbbar() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return hadbbar_->correctedJet("HAD", "B");
  }
  else if(jetCorrScheme_==2){
    return hadbbar_->correctedJet("HAD", "B");
  }
  else{
    return *hadbbar_;
  }
}

pat::Jet TtHadEvtSolution::getHadj() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return hadj_->correctedJet("HAD", "UDS");
  }
  else if(jetCorrScheme_==2){
    return hadj_->correctedJet("HAD", "UDS");
  }
  else{
    return *hadj_;
  }
}

pat::Jet TtHadEvtSolution::getHadk() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return hadk_->correctedJet("HAD", "UDS");
  }
  else if(jetCorrScheme_==2){
    return hadk_->correctedJet("HAD", "UDS");
  }
  else{
    return *hadk_;
  }
}

//-------------------------------------------
// get (un-)/calibrated reco objects
//-------------------------------------------
// By definition pq and b are the top quark, 
// jk and bbar the anti-top - check if it 
// makes sense ....
reco::Particle TtHadEvtSolution::getRecHadt() const 
{
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4()+this->getRecHadq().p4()+this->getRecHadb().p4());
}

reco::Particle TtHadEvtSolution::getRecHadtbar() const 
{
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadj().p4()+this->getRecHadk().p4()+this->getRecHadbbar().p4());
}

reco::Particle TtHadEvtSolution::getRecHadW_plus() const 
{
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadp().p4()+this->getRecHadq().p4());
}

reco::Particle TtHadEvtSolution::getRecHadW_minus() const 
{
  // FIXME: the charge from the genevent
  return reco::Particle(0,this->getRecHadj().p4()+this->getRecHadk().p4());
}

reco::Particle TtHadEvtSolution::getCalHadt() const 
{ 
  return reco::Particle(0,this->getCalHadp().p4()+this->getCalHadq().p4()+this->getCalHadb().p4()); 
}

reco::Particle TtHadEvtSolution::getCalHadtbar() const 
{ 
  return reco::Particle(0,this->getCalHadj().p4()+this->getCalHadk().p4()+this->getCalHadbbar().p4()); 
}

reco::Particle TtHadEvtSolution::getCalHadW_plus() const 
{ 
  return reco::Particle(0,this->getCalHadp().p4()+this->getCalHadq().p4()); 
}

reco::Particle TtHadEvtSolution::getCalHadW_minus() const 
{ 
  return reco::Particle(0,this->getCalHadj().p4()+this->getCalHadk().p4()); 
}

//-------------------------------------------
// get objects from kinematic fit
//-------------------------------------------  
reco::Particle TtHadEvtSolution::getFitHadt() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4()+this->getFitHadq().p4()+this->getFitHadb().p4());
}

reco::Particle TtHadEvtSolution::getFitHadtbar() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadj().p4()+this->getFitHadk().p4()+this->getFitHadbbar().p4());
}

reco::Particle TtHadEvtSolution::getFitHadW_plus() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadp().p4()+this->getFitHadq().p4());
}

reco::Particle TtHadEvtSolution::getFitHadW_minus() const 
{
  // FIXME: provide the correct charge from generated event
  return reco::Particle(0, this->getFitHadj().p4()+this->getFitHadk().p4());
}

//-------------------------------------------  
// get info on the outcome of the signal 
// selection LR
//-------------------------------------------  
double TtHadEvtSolution::getLRSignalEvtObsVal(unsigned int selObs) const 
{
  double val = -999.;
  for(size_t o=0; o<lrSignalEvtVarVal_.size(); o++){
    if(lrSignalEvtVarVal_[o].first == selObs) val = lrSignalEvtVarVal_[o].second;
  }
  return val;
}

//-------------------------------------------  
// get info on the outcome of the signal 
// selection LR
//-------------------------------------------  
double TtHadEvtSolution::getLRJetCombObsVal(unsigned int selObs) const 
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
void TtHadEvtSolution::setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt)
{
  if( !aGenEvt->isFullHadronic() ){ 
    edm::LogWarning( "TtGenEventNotFilled" ) << "genEvt is not fully hadronic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}

//-------------------------------------------
// methods to set the outcome of the different 
// jet combination methods
//-------------------------------------------
void TtHadEvtSolution::setLRJetCombObservables(const std::vector<std::pair<unsigned int, double> >& varval) 
{
  lrJetCombVarVal_.clear();
  for(size_t ijc = 0; ijc<varval.size(); ijc++) lrJetCombVarVal_.push_back(varval[ijc]);
}

//-------------------------------------------
// methods to set the outcome of the signal 
// selection LR
//-------------------------------------------
void TtHadEvtSolution::setLRSignalEvtObservables(const std::vector<std::pair<unsigned int, double> >& varval) 
{
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}
