//
// $Id: TtGenEvent.cc,v 1.27 2009/05/08 17:10:24 rwolf Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/CandUtils/interface/pdgIdUtils.h"
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
    if( abs(singleLepton()->pdgId())==TopDecayID::elecID ) type=WDecay::kElec;
    if( abs(singleLepton()->pdgId())==TopDecayID::muonID ) type=WDecay::kMuon;
    if( abs(singleLepton()->pdgId())==TopDecayID::tauID  ) type=WDecay::kTau;
  }
  return type;
}

std::pair<WDecay::LeptonType, WDecay::LeptonType>
TtGenEvent::fullLeptonicChannel() const 
{
  WDecay::LeptonType typeA=WDecay::kNone, typeB=WDecay::kNone;  
  if( isFullLeptonic() ){
    if( lepton() ){
      if( abs(lepton()->pdgId())==TopDecayID::elecID ) typeA=WDecay::kElec;
      if( abs(lepton()->pdgId())==TopDecayID::muonID ) typeA=WDecay::kMuon;
      if( abs(lepton()->pdgId())==TopDecayID::tauID  ) typeA=WDecay::kTau;      
    }
    if( leptonBar() ){
      if( abs(leptonBar()->pdgId())==TopDecayID::elecID ) typeB=WDecay::kElec;
      if( abs(leptonBar()->pdgId())==TopDecayID::muonID ) typeB=WDecay::kMuon;
      if( abs(leptonBar()->pdgId())==TopDecayID::tauID  ) typeB=WDecay::kTau;      
    }
  }
  return ( std::pair<WDecay::LeptonType,WDecay::LeptonType>(typeA, typeB) );
}

const reco::GenParticle* 
TtGenEvent::lepton(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand = 0;
  const reco::GenParticleCollection& partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isLepton(partsColl[i]) && partsColl[i].mother() &&
	abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])>0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonBar(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand = 0;
  const reco::GenParticleCollection& partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isLepton(partsColl[i]) && partsColl[i].mother() &&
	abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])<0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::singleLepton(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand = 0;
  if( isSemiLeptonic(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isLepton(partsColl[i]) && partsColl[i].mother() &&
	  abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrino(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isNeutrino(partsColl[i]) && partsColl[i].mother() &&
	abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])>0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrinoBar(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isNeutrino(partsColl[i]) && partsColl[i].mother() &&
	abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])<0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::singleNeutrino(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( isSemiLeptonic(excludeTauLeptons) ) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isNeutrino(partsColl[i]) && partsColl[i].mother() &&
	  abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayQuark(bool invert) const 
{
  const reco::GenParticle* cand=0;
  // catch W boson and check its daughters for a quarks; 
  // make sure the decay is semi-leptonic first; this 
  // only makes sense if taus are not excluded from the 
  // decision
  if( singleLepton(false) ){
    for(reco::GenParticleCollection::const_iterator w=parts_->begin(); w!=parts_->end(); ++w){
      if( abs( w->pdgId() )==TopDecayID::WID ){
	// make sure that the particle is a W daughter
	for(reco::GenParticle::const_iterator wd=w->begin(); wd!=w->end(); ++wd){ 
	  // make sure that the parton is a quark
	  if( abs(wd->pdgId())<TopDecayID::tID ){
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
TtGenEvent::hadronicDecayB(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::bID && 
	  reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayW(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::WID && 
          reco::flavour(singleLep) != -reco::flavour(partsColl[i])) { 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayTop(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayB(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::bID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayW(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::WID &&
          reco::flavour(singleLep) == - reco::flavour(partsColl[i])) { 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayTop(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for( unsigned int i = 0; i < partsColl.size(); ++i ){
      if( abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i]) ){
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

std::vector<const reco::GenParticle*> TtGenEvent::leptonicDecayTopRadiation(bool excludeTauLeptons) const{
  if( leptonicDecayTop(excludeTauLeptons) ){
    return (leptonicDecayTop(excludeTauLeptons)->pdgId()>0 ? radiatedGluons(TopDecayID::tID) : radiatedGluons(-TopDecayID::tID));
  }
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}

std::vector<const reco::GenParticle*> TtGenEvent::hadronicDecayTopRadiation(bool excludeTauLeptons) const{
  if( hadronicDecayTop(excludeTauLeptons) ){
    return (hadronicDecayTop(excludeTauLeptons)->pdgId()>0 ? radiatedGluons(TopDecayID::tID) : radiatedGluons(-TopDecayID::tID));
  }
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}
