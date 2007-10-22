#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

TopGenEvent::TopGenEvent(reco::CandidateRefProd& parts, std::vector<const reco::Candidate*> inits)
{
  parts_ = parts; 
  initPartons_= inits;
}

const reco::Candidate*
TopGenEvent::candidate(int id) const
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

int
TopGenEvent::numberOfLeptons() const
{
  int lep=0;
  const reco::CandidateCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isLepton(partsColl[i])&&(partsColl[i].status()==3)) {
      ++lep;
    }
  }  
  return lep;
}

int
TopGenEvent::numberOfBQuarks() const
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
TopGenEvent::singleLepton() const 
{
  const reco::Candidate* cand = 0;
  if (numberOfLeptons() == 1) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isLepton(partsColl[i])&&(partsColl[i].status()==3)) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::Candidate* 
TopGenEvent::singleNeutrino() const 
{
  const reco::Candidate* cand=0;
  if (numberOfLeptons()==1) {
    const reco::CandidateCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isNeutrino(partsColl[i])&&(partsColl[i].status()==3)) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

std::vector<const reco::Candidate*> 
TopGenEvent::lightQuarks(bool plusB) const 
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
