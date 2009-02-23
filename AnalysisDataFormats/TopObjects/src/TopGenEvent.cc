#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

TopGenEvent::TopGenEvent(reco::GenParticleRefProd& parts, reco::GenParticleRefProd& inits, int status)
{
  parts_ = parts; 
  initPartons_= inits;
  defaultStatus_ = status;
}

const reco::GenParticle*
TopGenEvent::candidate(int id) const
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (partsColl[i].pdgId()==id && partsColl[i].status()==defaultStatus_) {
      cand = &partsColl[i];
    }
  }  
  return cand;
}

void
TopGenEvent::dumpEventContent() const 
{
  edm::LogVerbatim log("topGenEvt");
  log << "\n"
      << "--------------------------------------\n"
      << "- Dump TopGenEvent Content           -\n"
      << "--------------------------------------\n";
  for (reco::GenParticleCollection::const_iterator part = parts_->begin(); 
       part<parts_->end(); ++part) {
    log << "pdgId:"  << std::setw(5)  << part->pdgId()     << ", "
	<< "mass:"   << std::setw(11) << part->p4().mass() << ", "
	<< "energy:" << std::setw(11) << part->energy()    << ", " 
	<< "pt:"     << std::setw(11) << part->pt()        << "\n"; 
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
TopGenEvent::numberOfLeptonsFromW() const
{
  int lep=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isLepton(partsColl[i])&&(partsColl[i].status()==3) && 
    // Leptons are coming from W decay 
     partsColl[i].mother() &&  abs(partsColl[i].mother()->pdgId())==24  ) {
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
   //depend if radiation qqbar are included or not
    if (abs(partsColl[i].pdgId())==5 && partsColl[i].status()==4) {// &&
      //abs(partsColl[i].mother()->pdgId())==6) {
      ++bq;
    }
  }  
  return bq;
}

int
TopGenEvent::numberOfBQuarksFromTop() const
{
  int bq=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
   //depend if radiation qqbar are included or not
    if (abs(partsColl[i].pdgId())==5 && partsColl[i].status()==4 && abs(partsColl[i].mother()->pdgId())==6) {
      ++bq;
    }
  }  
  return bq;
}

const reco::GenParticle* 
TopGenEvent::singleLepton() const 
{
  const reco::GenParticle* cand = 0;
  if (numberOfLeptonsFromW() == 1) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
     if (reco::isLepton(partsColl[i])&&(partsColl[i].status()==defaultStatus_)&&(partsColl[i].mother())&&(partsColl[i].mother()->pdgId())==24) {
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
  if (numberOfLeptonsFromW()==1) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isNeutrino(partsColl[i])&&(partsColl[i].status()==defaultStatus_)&&(partsColl[i].mother())&&(partsColl[i].mother()->pdgId())==24) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

std::vector<const reco::GenParticle*> 
TopGenEvent::lightQuarks(bool bIncluded) const 
{
  std::vector<const reco::GenParticle*> lightQuarks;
  reco::GenParticleCollection::const_iterator part = parts_->begin();
  for ( ; part < parts_->end(); ++part) {
    if( (bIncluded && abs(part->pdgId())==5) || (abs(part->pdgId())<5) && (part->status()==defaultStatus_) ) {
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      lightQuarks.push_back( part->clone() );
    }
  }  
  return lightQuarks;
}

const std::vector<const reco::GenParticle*> TopGenEvent::topRadiation() const{
  std::vector<const reco::GenParticle*> topRadiation;
  reco::GenParticleCollection::const_iterator part = parts_->begin();
  for ( ; part < parts_->end(); ++part) {
    if ( part->mother()!=NULL && part->mother()->pdgId()==6 && part->mother()->status()==3 && part->status()==2 ){
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      topRadiation.push_back( part->clone() );
    }
  }  
  return topRadiation;
}

const std::vector<const reco::GenParticle*> TopGenEvent::topBarRadiation() const{
  std::vector<const reco::GenParticle*> topBarRadiation;
  reco::GenParticleCollection::const_iterator part = parts_->begin();
  for ( ; part < parts_->end(); ++part) {
    if ( part->mother()!=NULL && part->mother()->pdgId()==-6 && part->mother()->status()==3 && part->status()==2 ){
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      topBarRadiation.push_back( part->clone() );
    }
  }  
  return topBarRadiation;
}

const std::vector<const reco::GenParticle*> TopGenEvent::ISR() const{
  std::vector<const reco::GenParticle*> ISR;
  reco::GenParticleCollection::const_iterator part = parts_->begin();
  for ( ; part < parts_->end(); ++part) {
    if ( part->pdgId()==21 && part->status()==3 && part->numberOfMothers()==0 ){
    // ISR are the only gluons status 3 in the collection & they don't have mother
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      ISR.push_back( part->clone() );
    }
  }  
  return ISR;
}
