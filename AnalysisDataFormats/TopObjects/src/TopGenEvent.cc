#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

TopGenEvent::TopGenEvent(reco::GenParticleRefProd& parts, reco::GenParticleRefProd& inits)
{
  parts_ = parts; 
  initPartons_= inits;
}

const reco::GenParticle*
TopGenEvent::candidate(int id) const
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].pdgId()==id) {
      cand = &partsColl[i];
    }
  }  
  return cand;
}

void
TopGenEvent::dumpEventContent() const 
{
  using std::cout;
  using std::endl;

  cout << endl;
  cout << "--------------------------------------" << endl;
  cout << "- Dump TopGenEvent Content           -" << endl;
  cout << "--------------------------------------" << endl;
  for (reco::GenParticleCollection::const_iterator part = parts_->begin(); 
       part<parts_->end(); ++part) {
    cout << "pdgId: "  << part->pdgId()     << " " 
	 << "mass: "   << part->p4().mass() << " "
	 << "energy: " << part->energy()    << " " 
	 << "pt: "     << part->pt() << endl; 
  }
}

int
TopGenEvent::numberOfLeptons() const
{
  int lep=0;
  const reco::GenParticleCollection & partsColl = *parts_;
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
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (abs(partsColl[i].pdgId())==5) {// &&
      //abs(partsColl[i].mother()->pdgId())==6) {
      ++bq;
    }
  }  
  return bq;
}

const reco::GenParticle* 
TopGenEvent::singleLepton() const 
{
  const reco::GenParticle* cand = 0;
  if (numberOfLeptons() == 1) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isLepton(partsColl[i])&&(partsColl[i].status()==3)) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TopGenEvent::singleNeutrino() const 
{
  const reco::GenParticle* cand=0;
  if (numberOfLeptons()==1) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isNeutrino(partsColl[i])&&(partsColl[i].status()==3)) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

std::vector<const reco::GenParticle*> 
TopGenEvent::lightQuarks(bool plusB) const 
{
  std::vector<const reco::GenParticle*> lightQuarks;
  reco::GenParticleCollection::const_iterator part = parts_->begin();
  for ( ; part < parts_->end(); ++part) {
    if( (plusB && abs(part->pdgId())==5) || (abs(part->pdgId())<5) ) {
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      lightQuarks.push_back( part->clone() );
    }
  }  
  return lightQuarks;
}
