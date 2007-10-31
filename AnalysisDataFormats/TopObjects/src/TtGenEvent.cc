//
// $Id: TtGenEvent.cc,v 1.17 2007/10/22 13:45:40 delaer Exp $
//
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

TtGenEvent::TtGenEvent()
{
}

TtGenEvent::TtGenEvent(reco::CandidateRefProd & parts, reco::CandidateRefProd & inits)
{
  parts_ = parts;
  initPartons_= inits;
}

TtGenEvent::~TtGenEvent()
{
}

const reco::Candidate* 
TtGenEvent::hadronicDecayQuark() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId()) < 5 && reco::flavour(partsColl[i])>0) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicDecayQuarkBar() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId()) < 5 && reco::flavour(partsColl[i])<0) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicDecayB() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 && 
	  reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicDecayW() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==24 && 
          reco::flavour(singleLep) != - reco::flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::hadronicDecayTop() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 &&
          reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicDecayB() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicDecayW() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==24 &&
          reco::flavour(singleLep) == - reco::flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TtGenEvent::leptonicDecayTop() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
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
    if (reco::isLepton(partsColl[i]) && reco::flavour(partsColl[i])>0&&(partsColl[i].status()==3)) {
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
    if (reco::isNeutrino(partsColl[i]) && reco::flavour(partsColl[i])>0&&(partsColl[i].status()==3)) {
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
    if (reco::isLepton(partsColl[i]) && reco::flavour(partsColl[i])<0&&(partsColl[i].status()==3)) {
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
    if (reco::isNeutrino(partsColl[i]) && reco::flavour(partsColl[i])<0&&(partsColl[i].status()==3)) {
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
    if (partsColl[i].mother() && reco::flavour(*(partsColl[i].mother()))<0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])>0) {
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
    if (partsColl[i].mother() && reco::flavour(*(partsColl[i].mother()))<0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])<0) {
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
    if (partsColl[i].mother() && reco::flavour(*(partsColl[i].mother()))>0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])>0) {
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
    if (partsColl[i].mother() && reco::flavour(*(partsColl[i].mother()))>0 &&
        abs(partsColl[i].pdgId())<5 && reco::flavour(partsColl[i])<0) {
      cand = &partsColl[i];
    }  
  }
  return cand;
}
