//
// $Id: TtSemiEvtSolution.cc,v 1.29 2013/04/19 22:13:23 wmtan Exp $
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

TtSemiEvtSolution::TtSemiEvtSolution() : 
  mcHyp_  ("ttSemiEvtMCHyp"), 
  recoHyp_("ttSemiEvtRecoHyp"),
  fitHyp_ ("ttSemiEvtFitHyp")
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
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if (jetCorrScheme_ == 1) return hadb_->correctedJet("HAD", "B"); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadb_->correctedJet("HAD", "B");
  else return *hadb_;
}

pat::Jet TtSemiEvtSolution::getHadp() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if (jetCorrScheme_ == 1) return hadp_->correctedJet("HAD", "UDS"); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadp_->correctedJet("HAD", "UDS");
  else return *hadp_;
}

pat::Jet TtSemiEvtSolution::getHadq() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if (jetCorrScheme_ == 1) return hadq_->correctedJet("HAD", "UDS"); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return hadq_->correctedJet("HAD", "UDS");
  else return *hadq_;
}

pat::Jet TtSemiEvtSolution::getLepb() const 
{
  // WARNING this is obsolete and only 
  // kept for backwards compatibility
  if (jetCorrScheme_ == 1) return lepb_->correctedJet("HAD", "B"); // calibrate jets according to MC truth
  else if (jetCorrScheme_ == 2) return lepb_->correctedJet("HAD", "B");
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
void TtSemiEvtSolution::setLRJetCombObservables(const std::vector<std::pair<unsigned int, double> >& varval) 
{
  lrJetCombVarVal_.clear();
  for(size_t ijc = 0; ijc<varval.size(); ijc++) lrJetCombVarVal_.push_back(varval[ijc]);
}

//-------------------------------------------  
// set the outcome of the signal selection LR
//-------------------------------------------  
void TtSemiEvtSolution::setLRSignalEvtObservables(const std::vector<std::pair<unsigned int, double> >& varval) 
{
  lrSignalEvtVarVal_.clear();
  for(size_t ise = 0; ise<varval.size(); ise++) lrSignalEvtVarVal_.push_back(varval[ise]);
}


void TtSemiEvtSolution::setupHyp() 
{

  AddFourMomenta addFourMomenta;

  recoHyp_.clearDaughters();
  recoHyp_.clearRoles();

  // Setup transient references
  reco::CompositeCandidate recHadt;
  reco::CompositeCandidate recLept;
  reco::CompositeCandidate recHadW;
  reco::CompositeCandidate recLepW;

  // Get refs to leaf nodes
  reco::ShallowClonePtrCandidate hadp( hadp_, hadp_->charge(), hadp_->p4(), hadp_->vertex() );
  reco::ShallowClonePtrCandidate hadq( hadq_, hadq_->charge(), hadq_->p4(), hadq_->vertex() );
  reco::ShallowClonePtrCandidate hadb( hadb_, hadb_->charge(), hadb_->p4(), hadb_->vertex() );
  reco::ShallowClonePtrCandidate lepb( lepb_, lepb_->charge(), lepb_->p4(), lepb_->vertex() );

  reco::ShallowClonePtrCandidate neutrino( neutrino_, neutrino_->charge(), neutrino_->p4(), neutrino_->vertex() );


//   JetCandRef hadp( hadp_->p4(), hadp_->charge(), hadp_->vertex());  hadp.setRef( hadp_ );
//   JetCandRef hadq( hadq_->p4(), hadq_->charge(), hadq_->vertex());  hadq.setRef( hadq_ );
//   JetCandRef hadb( hadb_->p4(), hadb_->charge(), hadb_->vertex());  hadb.setRef( hadb_ );
//   JetCandRef lepb( lepb_->p4(), lepb_->charge(), lepb_->vertex());  lepb.setRef( lepb_ );

//   METCandRef neutrino  ( neutrino_->p4(), neutrino_->charge(), neutrino_->vertex() ); neutrino.setRef( neutrino_ );



  recHadW.addDaughter( hadp,    "hadp" );
  recHadW.addDaughter( hadq,    "hadq" );

  addFourMomenta.set( recHadW );

  recHadt.addDaughter( hadb,    "hadb" );
  recHadt.addDaughter( recHadW, "hadW" );

  addFourMomenta.set( recHadt );
  
  recLepW.addDaughter( neutrino,"neutrino" );
  if ( getDecay() == "electron" ) {
    reco::ShallowClonePtrCandidate electron ( electron_, electron_->charge(), electron_->p4(), electron_->vertex() );
//     ElectronCandRef electron ( electron_->p4(), electron_->charge(), electron_->vertex() ); electron.setRef( electron_ );
    recLepW.addDaughter ( electron, "electron" );
  } else if ( getDecay() == "muon" ) {
    reco::ShallowClonePtrCandidate muon ( muon_, muon_->charge(),  muon_->p4(), muon_->vertex() );
//     MuonCandRef muon ( muon_->p4(), muon_->charge(), muon_->vertex() ); muon.setRef( muon_ );
    recLepW.addDaughter ( muon, "muon" );
  }

  addFourMomenta.set( recLepW );

  recLept.addDaughter( lepb,    "lepb" );
  recLept.addDaughter( recLepW,    "lepW" );

  addFourMomenta.set( recLept );

  recoHyp_.addDaughter( recHadt, "hadt" );
  recoHyp_.addDaughter( recLept, "lept" );

  addFourMomenta.set( recoHyp_ );


//   // Setup transient references
//   reco::CompositeCandidate fitHadt;
//   reco::CompositeCandidate fitLept;
//   reco::CompositeCandidate fitHadW;
//   reco::CompositeCandidate fitLepW;

//   // Get refs to leaf nodes
//   pat::Particle afitHadp = getFitHadp();
//   pat::Particle afitHadq = getFitHadq();
//   pat::Particle afitHadb = getFitHadb();
//   pat::Particle afitLepb = getFitLepb();
//   reco::ShallowClonePtrCandidate fitHadp( hadp_, afitHadp.charge(), afitHadp.p4(), afitHadp.vertex());
//   reco::ShallowClonePtrCandidate fitHadq( hadq_, afitHadq.charge(), afitHadq.p4(), afitHadq.vertex());
//   reco::ShallowClonePtrCandidate fitHadb( hadb_, afitHadb.charge(), afitHadb.p4(), afitHadb.vertex());
//   reco::ShallowClonePtrCandidate fitLepb( lepb_, afitLepb.charge(), afitLepb.p4(), afitLepb.vertex());

//   reco::ShallowClonePtrCandidate fitNeutrino  ( neutrino_, fitLepn_.charge(),  fitLepn_.p4(),  fitLepn_.vertex() );

//   fitHadW.addDaughter( fitHadp,    "hadp" );
//   fitHadW.addDaughter( fitHadq,    "hadq" );
//   fitHadt.addDaughter( fitHadb,    "hadb" );
//   fitHadt.addDaughter( fitHadW,    "hadW" );
  
//   fitLepW.addDaughter( fitNeutrino,"neutrino" );

//   if ( getDecay() == "electron" ) {
//     reco::ShallowClonePtrCandidate fitElectron ( electron_, electron_.charge(),  electron_.p4(), electron_.vertex() );
//     fitLepW.addDaughter ( fitElectron, "electron" );
//   } else if ( getDecay() == "muon" ) {
//     reco::ShallowClonePtrCandidate fitMuon ( muon_, muon_.charge(),  muon_.p4(), muon_.vertex() );
//     fitLepW.addDaughter ( fitMuon, "muon" );
//   }
//   fitLept.addDaughter( fitLepb,    "lepb" );
//   fitLept.addDaughter( fitLepW,    "lepW" );

//   fitHyp_.addDaughter( fitHadt,   "hadt" );
//   fitHyp_.addDaughter( fitLept,   "lept" );


  
  
}
