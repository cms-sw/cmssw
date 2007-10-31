//
// $Id: StGenEvent.cc,v 1.4 2007/10/19 12:31:53 rwolf Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"


StGenEvent::StGenEvent()
{
}

StGenEvent::StGenEvent(reco::CandidateRefProd & parts, reco::CandidateRefProd & inits)
{
  parts_ = parts;
  initPartons_= inits;
}

StGenEvent::~StGenEvent()
{
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
	  reco::flavour(singleLep)== - reco::flavour(partsColl[i])) { // ... but it should be the opposite!
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
	  reco::flavour(singleLep)== reco::flavour(partsColl[i])) { // ... but it should be the opposite!
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
          reco::flavour(singleLep) == - reco::flavour(partsColl[i])) { // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
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
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}
