#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

TtGenEvent::TtGenEvent()
{
}

TtGenEvent::TtGenEvent(reco::CandidateRefProd & candRefVec)
{
  parts_ = candRefVec;
}

TtGenEvent::~TtGenEvent()
{
}

int
TtGenEvent::numberOfLeptons() const
{
  int lep=0;
  const reco::CandidateCollection partsColl = *parts_;
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
//        abs(partsColl[i].mother()->pdgId())==6) {
      ++bq;
    }
  }  
  return bq;
}

const reco::Candidate*
TtGenEvent::candidate(int id) const
{
  const reco::Candidate* cand=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].pdgId()==id) {
      cand = &partsColl[i];
    }
  }  
  return cand;
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
	  flavour(singleLep)!=flavour(partsColl[i])) {
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
	  flavour(singleLep)==flavour(partsColl[i])) {
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
    if (partsColl[i].mother() && flavour(*(partsColl[i].mother()))<0
	&& abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])>0) {
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
    if (partsColl[i].mother() && flavour(*(partsColl[i].mother()))<0
	&& abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])<0) {
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
    if( partsColl[i].mother() && flavour(*(partsColl[i].mother()))>0
	&& abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])>0) {
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
    if( partsColl[i].mother() && flavour(*(partsColl[i].mother()))>0
	&& abs(partsColl[i].pdgId())<5 && flavour(partsColl[i])<0){
      cand = &partsColl[i];
    }  
  }
  return cand;
}
