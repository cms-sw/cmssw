//
// $Id: TtGenEvent.cc,v 1.24 2009/02/27 15:22:00 echabert Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


TtGenEvent::TtGenEvent(reco::GenParticleRefProd & parts, reco::GenParticleRefProd & inits)
{
  parts_ = parts;
  initPartons_= inits;
}

WDecay::LeptonType 
TtGenEvent::semiLeptonicChannel() const 
{
  WDecay::LeptonType type=WDecay::kNone;
  if( isSemiLeptonic() && singleLepton() ){
    if( fabs(singleLepton()->pdgId())==TopDecayID::elecID ) type=WDecay::kElec;
    if( fabs(singleLepton()->pdgId())==TopDecayID::muonID ) type=WDecay::kMuon;
    if( fabs(singleLepton()->pdgId())==TopDecayID::tauID  ) type=WDecay::kTau;
  }
  return type;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayQuark(bool invert) const 
{
  const reco::GenParticle* cand=0;
  // catch W boson and check its 
  // daughters for quarks
  if(singleLepton()) {
    for(reco::GenParticleCollection::const_iterator w=parts_->begin(); w!=parts_->end(); ++w){
      if( abs( w->pdgId() )==TopDecayID::WID ){
	// make sure that the particle is a w daughter
	for(reco::GenParticle::const_iterator wd=w->begin(); wd!=w->end(); ++wd){ 
	  // make sure that the parton is a quark
	  if( abs(wd->pdgId())<TopDecayID::tID){
	    if( invert ){ 
	      //treat ~q case
	      if( reco::flavour(*wd)<0 ){
		cand = dynamic_cast<const reco::GenParticle* > (&(*wd));
		if(cand == 0){
		  throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
		}
		break;
	      }
	    }
	    else{ 
	      //treat q case
	      if( reco::flavour(*wd)>0 ){
		cand = dynamic_cast<const reco::GenParticle* > (&(*wd));
		if(cand == 0){
		  throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
		}
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayB() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton();
    for(unsigned int i = 0; i < partsColl.size(); ++i) {
      if(abs(partsColl[i].pdgId())==TopDecayID::bID &&
	  reco::flavour(singleLep)==reco::flavour(partsColl[i])){
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayW() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for(unsigned int i = 0; i < partsColl.size(); ++i) {
      if(abs(partsColl[i].pdgId())==TopDecayID::WID &&
          reco::flavour(singleLep) != -reco::flavour(partsColl[i])){ 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayTop() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for(unsigned int i = 0; i < partsColl.size(); ++i){
      if(abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayB() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for(unsigned int i = 0; i < partsColl.size(); ++i){
      if(abs(partsColl[i].pdgId())==TopDecayID::bID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayW() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for(unsigned int i = 0; i < partsColl.size(); ++i){
      if(abs(partsColl[i].pdgId())==TopDecayID::WID &&
          reco::flavour(singleLep) == -reco::flavour(partsColl[i])) { 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayTop() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for(unsigned int i = 0; i < partsColl.size(); ++i){
      if(abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::lepton() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for(unsigned int i = 0; i < partsColl.size(); ++i){ 
    if(reco::isLepton(partsColl[i]) && partsColl[i].mother() && 
       abs(partsColl[i].mother()->pdgId())==TopDecayID::WID){
      if( reco::flavour(partsColl[i])>0 ) {
	cand = &partsColl[i];
      } 
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrino() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i){
    if(reco::isNeutrino(partsColl[i]) && partsColl[i].mother() && 
       abs(partsColl[i].mother()->pdgId())==TopDecayID::WID){
      if( reco::flavour(partsColl[i])>0 ) {
	cand = &partsColl[i];
      }
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for(unsigned int i = 0; i < partsColl.size(); ++i){ 
    if(reco::isLepton(partsColl[i]) && partsColl[i].mother() && 
       abs(partsColl[i].mother()->pdgId())==TopDecayID::WID){
      if( reco::flavour(partsColl[i])<0 ) {
	cand = &partsColl[i];
      } 
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrinoBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i){
    if(reco::isNeutrino(partsColl[i]) && partsColl[i].mother() && 
       abs(partsColl[i].mother()->pdgId())==TopDecayID::WID){
      if( reco::flavour(partsColl[i])>0 ) {
	cand = &partsColl[i];
      }
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::lightQFromTopBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if(partsColl[i].mother() && partsColl[i].mother()->pdgId()==-TopDecayID::WID &&
       abs(partsColl[i].pdgId())<TopDecayID::bID && reco::flavour(partsColl[i])>0){
      cand = &partsColl[i];
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::lightQBarFromTopBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if(partsColl[i].mother() && partsColl[i].mother()->pdgId()==-TopDecayID::WID &&
       abs(partsColl[i].pdgId())<TopDecayID::bID && reco::flavour(partsColl[i])<0){
      cand = &partsColl[i];
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::lightQFromTop() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if(partsColl[i].mother() && partsColl[i].mother()->pdgId()==TopDecayID::WID &&
       abs(partsColl[i].pdgId())<TopDecayID::bID && reco::flavour(partsColl[i])>0){
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::lightQBarFromTop() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if(partsColl[i].mother() && partsColl[i].mother()->pdgId()==TopDecayID::WID &&
       abs(partsColl[i].pdgId())<TopDecayID::bID && reco::flavour(partsColl[i])<0){
      cand = &partsColl[i];
    }  
  }
  return cand;
}

std::vector<const reco::GenParticle*> TtGenEvent::leptonicDecayTopRadiation() const{
  if(leptonicDecayTop()){
    return (hadronicDecayTop()->pdgId()>0 ? radiatedGluons(TopDecayID::tID) : radiatedGluons(-TopDecayID::tID));
  }
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}

std::vector<const reco::GenParticle*> TtGenEvent::hadronicDecayTopRadiation() const{
  if(hadronicDecayTop()){
    return (hadronicDecayTop()->pdgId()>0 ? radiatedGluons(TopDecayID::tID) : radiatedGluons(-TopDecayID::tID));
  }
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}
