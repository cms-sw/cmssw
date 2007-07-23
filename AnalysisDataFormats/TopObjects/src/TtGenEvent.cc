//
// $Id: TtGenEvent.cc,v 1.10 2007/07/20 14:36:45 rwolf Exp $
//
#include "FWCore/Utilities/interface/EDMException.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

TtGenEvent::TtGenEvent()
{
}

TtGenEvent::TtGenEvent(reco::CandidateRefProd & parts, std::vector<const reco::Candidate*> inits)
{
  parts_ = parts;
  initPartons_= inits;
}

TtGenEvent::~TtGenEvent()
{
}

int
TtGenEvent::numberOfLeptons() const
{
  int lep=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (isLepton(partsColl[i])) {
      ++lep;
    }
  }  
  return lep;
}

int
TtGenEvent::numberOfBQuarks() const
{
  int bq=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (abs(partsColl[i].pdgId())==5) {// &&
      //abs(partsColl[i].mother()->pdgId())==6) {
      ++bq;
    }
  }  
  return bq;
}

const reco::Candidate* 
TtGenEvent::singleLepton() const 
{
  const reco::Candidate* cand = 0;
  if (numberOfLeptons() == 1) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (isLepton(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::singleNeutrino() const 
{
  const reco::Candidate* cand=0;
  if (numberOfLeptons()==1) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (isNeutrino(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicQuark() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId()) < 5 && flavour(partsColl[i])>0) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicQuarkBar() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId()) < 5 && flavour(partsColl[i])<0) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicB() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::Candidate & singleLep = *singleLepton();
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 && 
	  flavour(singleLep)==flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicW() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::Candidate & singleLep = *singleLepton();
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==24 && 
          flavour(singleLep) != - flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicTop() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::Candidate & singleLep = *singleLepton();
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 &&
          flavour(singleLep)==flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicB() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::Candidate & singleLep = *singleLepton();
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 &&
          flavour(singleLep)!=flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicW() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::Candidate & singleLep = *singleLepton();
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==24 &&
          flavour(singleLep) == - flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicTop() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::Candidate & singleLep = *singleLepton();
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 &&
          flavour(singleLep)!=flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::lepton() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (isLepton(partsColl[i]) && flavour(partsColl[i])>0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::neutrino() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (isNeutrino(partsColl[i]) && flavour(partsColl[i])>0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonBar() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (isLepton(partsColl[i]) && flavour(partsColl[i])<0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::neutrinoBar() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (isNeutrino(partsColl[i]) && flavour(partsColl[i])<0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromTop() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother() && flavour(*(partsColl[i].mother()))<0 &&
        abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])>0) {
      cand = &partsColl[i];
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromTopBar() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother() && flavour(*(partsColl[i].mother()))<0 &&
        abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])<0) {
      cand = &partsColl[i];
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromAntiTop() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother() && flavour(*(partsColl[i].mother()))>0 &&
        abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])>0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::quarkFromAntiTopBar() const 
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].mother() && flavour(*(partsColl[i].mother()))>0 &&
        abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])<0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}

std::vector<const reco::Candidate*> 
TtGenEvent::lightQuarks(bool plusB=false) const 
{
  std::vector<const reco::Candidate*> lightQuarks;
  reco::CandidateCollection::const_iterator part = parts_->begin();
  for ( ; part < parts_->end(); ++part) {
    if( (plusB && abs(part->pdgId())==5) || (abs(part->pdgId())<5) ) {
      if( dynamic_cast<const reco::GenParticleCandidate*>( &(*part) ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticleCandidate" );
      }
      lightQuarks.push_back( part->clone() );
    }
  }  
  return lightQuarks;
}

