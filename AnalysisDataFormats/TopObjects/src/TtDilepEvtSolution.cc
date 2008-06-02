//
// $Id: TtDilepEvtSolution.cc,v 1.16 2008/02/15 12:10:53 rwolf Exp $
//

#include "PhysicsTools/Utilities/interface/DeltaR.h"
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
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return jetB_->mcFlavCorrJet();
  }
  else if(jetCorrScheme_==2){
    return jetB_->bCorrJet();
  }
  else{
    return *jetB_;
  }
}

pat::Jet TtDilepEvtSolution::getJetBbar() const 
{
  if(jetCorrScheme_==1){
    //jet calibrated according to MC truth
    return jetBbar_->mcFlavCorrJet();
  }
  else if(jetCorrScheme_==2){
    return jetBbar_->bCorrJet();
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
  distance += DeltaR<reco::Particle,reco::GenParticle>()(getCalJetB(),*getGenB());
  distance += DeltaR<reco::Particle,reco::GenParticle>()(getCalJetBbar(),*getGenBbar());
  return distance;
}

double TtDilepEvtSolution::getLeptonResidual() const
{
  double distance = 0.;
  if(!getGenLepp() || !getGenLepm()) return distance;
  if(getWpDecay()=="electron")
    distance += DeltaR<reco::Particle,reco::GenParticle>()(getElectronp(),*getGenLepp());
  else if(getWpDecay()=="muon")
    distance += DeltaR<reco::Particle,reco::GenParticle>()(getMuonp(),*getGenLepp());
  else if(getWpDecay()=="tau")
    distance += DeltaR<reco::Particle,reco::GenParticle>()(getTaup(),*getGenLepp());
  if(getWmDecay()=="electron")
    distance += DeltaR<reco::Particle,reco::GenParticle>()(getElectronm(),*getGenLepm());
  else if(getWmDecay()=="muon")
    distance += DeltaR<reco::Particle,reco::GenParticle>()(getMuonm(),*getGenLepm());
  else if(getWmDecay()=="tau")
    distance += DeltaR<reco::Particle,reco::GenParticle>()(getTaum(),*getGenLepm());
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
void TtDilepEvtSolution::setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval) 
{
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}
