//
// $Id$
//

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


/// constructor
TtDilepEvtSolution::TtDilepEvtSolution() {
  wpDecay_ = "NotDefined";
  wmDecay_ = "NotDefined";
  bestSol_ = false;
}


/// destructor
TtDilepEvtSolution::~TtDilepEvtSolution() {
}


// members to get original TopObjects 
TopJet      TtDilepEvtSolution::getJetB() const      { return *jetB_; }
TopJet      TtDilepEvtSolution::getJetBbar() const   { return *jetBbar_; }
TopElectron TtDilepEvtSolution::getElectronp() const { return *elecp_; }
TopElectron TtDilepEvtSolution::getElectronm() const { return *elecm_; }
TopMuon     TtDilepEvtSolution::getMuonp() const     { return *muonp_; }
TopMuon     TtDilepEvtSolution::getMuonm() const     { return *muonm_; }
TopMET      TtDilepEvtSolution::getMET() const       { return *met_; }


// methods to get the MC matched particles
// FIXME: provide defaults if the genevent is invalid
const TtGenEvent &      TtDilepEvtSolution::getGenEvent() const { return *theGenEvt_; }
const reco::Candidate * TtDilepEvtSolution::getGenT() const    { return theGenEvt_->top(); }
const reco::Candidate * TtDilepEvtSolution::getGenWp() const   { return theGenEvt_->wBar(); }
const reco::Candidate * TtDilepEvtSolution::getGenB() const    { return theGenEvt_->b(); }
const reco::Candidate * TtDilepEvtSolution::getGenLepp() const { return theGenEvt_->leptonBar(); }
const reco::Candidate * TtDilepEvtSolution::getGenN() const    { return theGenEvt_->neutrino(); }
const reco::Candidate * TtDilepEvtSolution::getGenTbar() const { return theGenEvt_->topBar(); }
const reco::Candidate * TtDilepEvtSolution::getGenWm() const   { return theGenEvt_->w(); }
const reco::Candidate * TtDilepEvtSolution::getGenBbar() const { return theGenEvt_->bBar(); }
const reco::Candidate * TtDilepEvtSolution::getGenLepm() const { return theGenEvt_->lepton(); }
const reco::Candidate * TtDilepEvtSolution::getGenNbar() const { return theGenEvt_->neutrinoBar(); }


// methods to explicitly get reconstructed and calibrated objects 
TopJetType TtDilepEvtSolution::getRecJetB() const    { return this->getJetB().getRecJet(); }
TopJet     TtDilepEvtSolution::getCalJetB() const    { return this->getJetB(); }
TopJetType TtDilepEvtSolution::getRecJetBbar() const { return this->getJetBbar().getRecJet(); }
TopJet     TtDilepEvtSolution::getCalJetBbar() const { return this->getJetBbar(); }


// method to set the generated event
void TtDilepEvtSolution::setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt) {
  if( !aGenEvt->isFullLeptonic() ) {
    edm::LogWarning( "TtGenEventNotFilled" ) << "genEvt is not di-leptonic; TtGenEvent is not filled";
    return;
  }
  theGenEvt_ = edm::RefProd<TtGenEvent>(aGenEvt);
}


// methods to set the basic TopObjects
void TtDilepEvtSolution::setB(const edm::Handle<std::vector<TopJet> > & jh, int i)              { jetB_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtDilepEvtSolution::setBbar(const edm::Handle<std::vector<TopJet> > & jh, int i)           { jetBbar_ = edm::Ref<std::vector<TopJet> >(jh, i); }
void TtDilepEvtSolution::setMuonp(const edm::Handle<std::vector<TopMuon> > & mh, int i)         { muonp_ = edm::Ref<std::vector<TopMuon> >(mh, i); wpDecay_ = "muon"; }
void TtDilepEvtSolution::setMuonm(const edm::Handle<std::vector<TopMuon> > & mh, int i)         { muonm_ = edm::Ref<std::vector<TopMuon> >(mh, i); wmDecay_ = "muon"; }
void TtDilepEvtSolution::setElectronp(const edm::Handle<std::vector<TopElectron> > & eh, int i) { elecp_ = edm::Ref<std::vector<TopElectron> >(eh, i); wpDecay_ = "electron"; }
void TtDilepEvtSolution::setElectronm(const edm::Handle<std::vector<TopElectron> > & eh, int i) { elecm_ = edm::Ref<std::vector<TopElectron> >(eh, i); wmDecay_ = "electron"; }
void TtDilepEvtSolution::setMET(const edm::Handle<std::vector<TopMET> > & nh, int i)            { met_ = edm::Ref<std::vector<TopMET> >(nh, i); }


// miscellaneous methods
void TtDilepEvtSolution::setBestSol(bool bs)       { bestSol_ = bs; }
void TtDilepEvtSolution::setRecTopMass(double j)   { topmass_ = j; }
void TtDilepEvtSolution::setRecWeightMax(double j) { weightmax_ = j; }

