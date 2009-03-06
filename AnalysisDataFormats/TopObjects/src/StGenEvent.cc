//
// $Id: StGenEvent.cc,v 1.7 2008/02/15 12:10:51 rwolf Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

StGenEvent::StGenEvent()
{
}

StGenEvent::StGenEvent(reco::GenParticleRefProd & parts, reco::GenParticleRefProd & inits)
{
  parts_ = parts;
  initPartons_= inits;
}

StGenEvent::~StGenEvent()
{
}

const reco::GenParticle* 
StGenEvent::decayB() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::bID && 
	  reco::flavour(singleLep)== - reco::flavour(partsColl[i])) { 
	// ... but it should be the opposite!
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
StGenEvent::associatedB() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::bID && 
	  reco::flavour(singleLep)== reco::flavour(partsColl[i])) { 
	// ... but it should be the opposite!
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
StGenEvent::singleW() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::WID &&
          reco::flavour(singleLep) == - reco::flavour(partsColl[i])){ 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
StGenEvent::singleTop() const 
{
  const reco::GenParticle* cand=0;
  if (singleLepton()) {
    const reco::GenParticleCollection & partsColl = *parts_;
    const reco::GenParticle & singleLep = *singleLepton();
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}
