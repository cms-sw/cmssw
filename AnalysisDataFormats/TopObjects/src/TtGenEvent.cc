//
// $Id: TtGenEvent.cc,v 1.22.2.2 2009/01/23 02:12:30 rwolf Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

TtGenEvent::TtGenEvent()
{
}

TtGenEvent::TtGenEvent(reco::GenParticleRefProd & parts, reco::GenParticleRefProd & inits, int status)
{
  parts_ = parts;
  initPartons_= inits;
  defaultStatus_ = status;
}

TtGenEvent::~TtGenEvent()
{
}

TtGenEvent::LeptonType 
TtGenEvent::semiLeptonicChannel() const 
{
  LeptonType type=kNone;
  if( isSemiLeptonic() && singleLepton() ){
    if( fabs(singleLepton()->pdgId())==11 ) type=kElec;
    if( fabs(singleLepton()->pdgId())==13 ) type=kMuon;
    if( fabs(singleLepton()->pdgId())==15 ) type=kTau;
  }
  return type;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayQuark(bool invert) const 
{
  const reco::GenParticle* cand=0;
  //catch W boson and check for its daughters for a quark
  if (singleLepton()) {
    for(reco::GenParticleCollection::const_iterator w=parts_->begin(); w!=parts_->end(); ++w){
      if( w->status()==defaultStatus_ && abs( w->pdgId() )==TopDecayID::WID ){
	// make sure that the particle is a w daughter
	for(reco::GenParticle::const_iterator wd=w->begin(); wd!=w->end(); ++wd){ 
	  // make sure that the parton is a quark
	  if( wd->status()==defaultStatus_ && abs(wd->pdgId())<TopDecayID::tID){
	    if( invert ){ //treat ~q case
	      if( reco::flavour(*wd)<0 ){
		cand = dynamic_cast<const reco::GenParticle* > (&(*wd));
		if(cand == 0){
		  throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
		}
		break;
	      }
	    }
	    else{         //treat q case
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
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 &&  partsColl[i].status()==defaultStatus_ &&
	  reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
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
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==24 && partsColl[i].status()==defaultStatus_ &&
          reco::flavour(singleLep) != - reco::flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
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
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 && partsColl[i].status()==defaultStatus_ &&
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
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 && partsColl[i].status()==defaultStatus_ &&
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
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==24 && partsColl[i].status()==defaultStatus_ &&
          reco::flavour(singleLep) == - reco::flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
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
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 && partsColl[i].status()==defaultStatus_ &&
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
  for (unsigned int i = 0; i < partsColl.size(); ++i) { // particles keep status 3 in decaySubset
    if (reco::isLepton(partsColl[i]) && reco::flavour(partsColl[i])>0&&(partsColl[i].status()==3)) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrino() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) { // particles keep status 3 in decaySubset
    if (reco::isNeutrino(partsColl[i]) && reco::flavour(partsColl[i])>0&&(partsColl[i].status()==3)) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) { // particles keep status 3 in decaySubset
    if (reco::isLepton(partsColl[i]) && reco::flavour(partsColl[i])<0&&(partsColl[i].status()==3)) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrinoBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) { // particles keep status 3 in decaySubset
    if (reco::isNeutrino(partsColl[i]) && reco::flavour(partsColl[i])<0&&(partsColl[i].status()==3)) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::quarkFromTop() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother(0) && reco::flavour(*(partsColl[i].mother(0)))<0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])>0 && partsColl[i].status()==defaultStatus_) {
      cand = &partsColl[i];
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::quarkFromTopBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother(0) && reco::flavour(*(partsColl[i].mother(0)))<0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])<0 && partsColl[i].status()==defaultStatus_) {
      cand = &partsColl[i];
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::quarkFromAntiTop() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother(0) && reco::flavour(*(partsColl[i].mother(0)))>0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])>0 && partsColl[i].status()==defaultStatus_) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::quarkFromAntiTopBar() const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother(0) && reco::flavour(*(partsColl[i].mother(0)))>0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])<0 && partsColl[i].status()==defaultStatus_) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

std::vector<const reco::GenParticle*> TtGenEvent::leptonicDecayTopRadiation() const{
  if(leptonicDecayTop() && leptonicDecayTop()->pdgId()==6) return (topRadiation());
  if(leptonicDecayTop() && leptonicDecayTop()->pdgId()==-6) return (topBarRadiation());
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}

std::vector<const reco::GenParticle*> TtGenEvent::hadronicDecayTopRadiation() const{
  if(hadronicDecayTop() && hadronicDecayTop()->pdgId()==6) return (topRadiation());
  if(hadronicDecayTop() && hadronicDecayTop()->pdgId()==-6) return (topBarRadiation());
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}
