//
// $Id: TtDilepEvtSolution.cc,v 1.22 2013/04/19 22:13:23 wmtan Exp $
//

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"

TtDilepEvtSolution::TtDilepEvtSolution() 
{
  jetCorrScheme_ = 0;
  wpDecay_ = "NotDefined";
  wmDecay_ = "NotDefined";
  bestSol_ = false;
  topmass_ = 0.;
  weightmax_ = 0.;
}

TtDilepEvtSolution::~TtDilepEvtSolution() 
{
}

//-------------------------------------------
// get calibrated base objects 
//-------------------------------------------
pat::Jet TtDilepEvtSolution::getJetB() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return jetB_->correctedJet("HAD", "B");
  }
  else if(jetCorrScheme_==2){
    return jetB_->correctedJet("HAD", "B");
  }
  else{
    return *jetB_;
  }
}

pat::Jet TtDilepEvtSolution::getJetBbar() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return jetBbar_->correctedJet("HAD", "B");
  }
  else if(jetCorrScheme_==2){
    return jetBbar_->correctedJet("HAD", "B");
  }
  else{
    return *jetBbar_;
  }
}

//-------------------------------------------
// returns the 4-vector of the positive 
// lepton, with the charge and the pdgId
//-------------------------------------------
reco::Particle TtDilepEvtSolution::getLeptPos() const 
{
  reco::Particle p;
  if(wpDecay_ == "muon"){
    p = reco::Particle(+1, getMuonp().p4() );
    p.setPdgId(-11);
  }
  if(wpDecay_ == "electron"){
    p = reco::Particle(+1, getElectronp().p4() );
    p.setPdgId(-13);
  }
  if(wmDecay_ == "tau"){
    p = reco::Particle(+1, getTaup().p4() );
    p.setPdgId(-15);
  }
  return p;
}

//-------------------------------------------
// miscellaneous
//-------------------------------------------
double TtDilepEvtSolution::getJetResidual() const
{
  double distance = 0.;
  if(!getGenB() || !getGenBbar()) return distance;
  distance += reco::deltaR(getCalJetB(),*getGenB());
  distance += reco::deltaR(getCalJetBbar(),*getGenBbar());
  return distance;
}

double TtDilepEvtSolution::getLeptonResidual() const
{
  double distance = 0.;
  if(!getGenLepp() || !getGenLepm()) return distance;
  if(getWpDecay()=="electron")
    distance += reco::deltaR(getElectronp(),*getGenLepp());
  else if(getWpDecay()=="muon")
    distance += reco::deltaR(getMuonp(),*getGenLepp());
  else if(getWpDecay()=="tau")
    distance += reco::deltaR(getTaup(),*getGenLepp());
  if(getWmDecay()=="electron")
    distance += reco::deltaR(getElectronm(),*getGenLepm());
  else if(getWmDecay()=="muon")
    distance += reco::deltaR(getMuonm(),*getGenLepm());
  else if(getWmDecay()=="tau")
    distance += reco::deltaR(getTaum(),*getGenLepm());
  return distance;
}

//-------------------------------------------
// returns the 4-vector of the negative 
// lepton, with the charge and the pdgId
//-------------------------------------------
reco::Particle TtDilepEvtSolution::getLeptNeg() const 
{
  reco::Particle p;
  if(wmDecay_ == "electron"){
    p = reco::Particle(-1, getElectronm().p4() );
    p.setPdgId(11);
  }
  if(wmDecay_ == "muon"){
    p = reco::Particle(-1, getMuonm().p4() );
    p.setPdgId(13);
  }
  if(wmDecay_ == "tau"){
    p = reco::Particle(-1, getTaum().p4() );
    p.setPdgId(15);
  }
  return p;
}

//-------------------------------------------
// get info on the outcome of the signal 
//selection LR
//-------------------------------------------
double TtDilepEvtSolution::getLRSignalEvtObsVal(unsigned int selObs) const 
{
  double val = -999.;
  for(size_t i=0; i<lrSignalEvtVarVal_.size(); i++){
    if(lrSignalEvtVarVal_[i].first == selObs) val = lrSignalEvtVarVal_[i].second;
  }
  return val;
}

//-------------------------------------------
// set the generated event
//-------------------------------------------
void TtDilepEvtSolution::setGenEvt(const edm::Handle<TtGenEvent>& aGenEvt) {
  if( !aGenEvt->isFullLeptonic() ){
    edm::LogInfo( "TtGenEventNotFilled" ) << "genEvt is not di-leptonic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}

//-------------------------------------------
// set the outcome of the signal selection LR
//-------------------------------------------
void TtDilepEvtSolution::setLRSignalEvtObservables(const std::vector<std::pair<unsigned int, double> >& varval) 
{
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}
