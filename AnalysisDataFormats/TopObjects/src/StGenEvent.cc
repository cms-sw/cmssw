//
// $Id: StGenEvent.cc,v 1.2.2.1 2007/08/02 16:28:33 giamman Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"


StGenEvent::StGenEvent()
{
}

StGenEvent::StGenEvent(reco::CandidateRefProd & parts, std::vector<const reco::Candidate*> inits)
{
  parts_ = parts;
  initPartons_= inits;
}

StGenEvent::~StGenEvent()
{
}

int
StGenEvent::numberOfLeptons() const
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
StGenEvent::numberOfBQuarks() const
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
StGenEvent::singleLepton() const 
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
StGenEvent::singleNeutrino() const 
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
StGenEvent::decayB() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 && 
	  flavour(singleLep)== - flavour(partsColl[i])) { // ... but it should be the opposite!
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
StGenEvent::associatedB() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (abs(partsColl[i].pdgId())==5 && 
	  flavour(singleLep)== flavour(partsColl[i])) { // ... but it should be the opposite!
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
StGenEvent::singleW() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
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
StGenEvent::singleTop() const 
{
  const reco::Candidate* cand=0;
  if (singleLepton()) {
    const reco::CandidateCollection & partsColl = *parts_;
    const reco::Candidate & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==6 &&
          flavour(singleLep)!=flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

std::vector<const reco::Candidate*> 
StGenEvent::lightQuarks(bool plusB) const 
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

