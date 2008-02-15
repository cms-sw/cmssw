//
// $Id: TtGenEvent.cc,v 1.19 2008/01/25 13:34:29 vadler Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

TtGenEvent::TtGenEvent()
{
}

TtGenEvent::TtGenEvent(reco::GenParticleRefProd & parts, reco::GenParticleRefProd & inits)
{
  parts_ = parts;
  initPartons_= inits;
}

TtGenEvent::~TtGenEvent()
{
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayQuark() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId()) < 5 && reco::flavour(partsColl[i])>0) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayQuarkBar() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId()) < 5 && reco::flavour(partsColl[i])<0) {
        cand = &partsColl[i];
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
      if (abs(partsColl[i].pdgId())==5 && 
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
      if (abs(partsColl[i].pdgId())==24 && 
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
      if (abs(partsColl[i].pdgId())==6 &&
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
      if (abs(partsColl[i].pdgId())==5 &&
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
      if (abs(partsColl[i].pdgId())==24 &&
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
      if (abs(partsColl[i].pdgId())==6 &&
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
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
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
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
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
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
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
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
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
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])>0) {
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
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])<0) {
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
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])>0) {
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
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])<0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}
