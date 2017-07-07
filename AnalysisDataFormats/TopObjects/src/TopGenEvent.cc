#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/CandUtils/interface/pdgIdUtils.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

/// default contructor
TopGenEvent::TopGenEvent(reco::GenParticleRefProd& decaySubset, reco::GenParticleRefProd& initSubset)
{
  parts_ = decaySubset; 
  initPartons_= initSubset;
}

const reco::GenParticle*
TopGenEvent::candidate(int id, unsigned int parentId) const
{
  const reco::GenParticle* cand=nullptr;
  const reco::GenParticleCollection & partsColl = *parts_;
  for( unsigned int i = 0; i < partsColl.size(); ++i ) {
    if( partsColl[i].pdgId()==id ){
      if(parentId==0?true:partsColl[i].mother()&&std::abs(partsColl[i].mother()->pdgId())==(int)parentId){
	cand = &partsColl[i];
      }
    }
  }  
  return cand;
}

void
TopGenEvent::print() const 
{
  edm::LogVerbatim log("TopGenEvent");
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
TopGenEvent::numberOfLeptons(bool fromWBoson) const
{
  int lep=0;
  const reco::GenParticleCollection& partsColl = *parts_;
  for(unsigned int i = 0; i < partsColl.size(); ++i) {
    if(reco::isLepton(partsColl[i])) {
      if(fromWBoson){
	if(partsColl[i].mother() &&  std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID){
	  ++lep;
	}
      }
      else{
	++lep;
      }
    }
  }  
  return lep;
}

int
TopGenEvent::numberOfLeptons(WDecay::LeptonType typeRestriction, bool fromWBoson) const
{
  int leptonType=-1;
  switch(typeRestriction){
    // resolve whether or not there is
    // any restriction in lepton types
  case WDecay::kElec: 
    leptonType=TopDecayID::elecID;
    break;
  case WDecay::kMuon: 
    leptonType=TopDecayID::muonID;
    break;
  case WDecay::kTau: 
    leptonType=TopDecayID::tauID;
    break;
  case WDecay::kNone:
    break;
  }
  int lep=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for(unsigned int i = 0; i < partsColl.size(); ++i) {
    if(fromWBoson){
      // restrict to particles originating from the W boson
      if( !(partsColl[i].mother() &&  std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) ){
	continue;
      }
    }
    if(leptonType>0){
      // in case of lepton type restriction
      if( std::abs(partsColl[i].pdgId())==leptonType ){
	++lep;
      }
    }
    else{
      // take any lepton type into account else
      if( reco::isLepton(partsColl[i]) ){
	++lep;
      }
    }
  }  
  return lep;
}

int
TopGenEvent::numberOfBQuarks(bool fromTopQuark) const
{
  int bq=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
   //depend if radiation qqbar are included or not
    if(std::abs(partsColl[i].pdgId())==TopDecayID::bID){
      if(fromTopQuark){
	if(partsColl[i].mother() &&  std::abs(partsColl[i].mother()->pdgId())==TopDecayID::tID){
	  ++bq;
	}
      }
      else{
	++bq;
      }
    }
  }  
  return bq;
}

std::vector<const reco::GenParticle*> 
TopGenEvent::topSisters() const
{
  std::vector<const reco::GenParticle*> sisters;
  for(reco::GenParticleCollection::const_iterator part = parts_->begin(); part<parts_->end(); ++part){
    if( part->numberOfMothers()==0 && std::abs(part->pdgId())!= TopDecayID::tID){
      // choose top sister which do not have a 
      // mother and are whether top nor anti-top 
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == nullptr){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      sisters.push_back( part->clone() );
    }
  }  
  return sisters;
}

const reco::GenParticle*
TopGenEvent::daughterQuarkOfTop(bool invertCharge) const
{
  const reco::GenParticle* cand=nullptr;
  for(reco::GenParticleCollection::const_iterator top = parts_->begin(); top<parts_->end(); ++top){
    if( top->pdgId()==(invertCharge?-TopDecayID::tID:TopDecayID::tID) ){
      for(reco::GenParticle::const_iterator quark = top->begin(); quark<top->end(); ++quark){
	if( std::abs(quark->pdgId())<= TopDecayID::bID ){
	  cand = dynamic_cast<const reco::GenParticle* > (&(*quark));
	  if(cand == nullptr){
	    throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
	  }
	  break;
	}
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TopGenEvent::daughterQuarkOfWPlus(bool invertQuarkCharge, bool invertBosonCharge) const 
{
  const reco::GenParticle* cand=nullptr;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if(partsColl[i].mother() && partsColl[i].mother()->pdgId()==(invertBosonCharge?-TopDecayID::WID:TopDecayID::WID) &&
       std::abs(partsColl[i].pdgId())<=TopDecayID::bID && (invertQuarkCharge?reco::flavour(partsColl[i])<0:reco::flavour(partsColl[i])>0)){
      cand = &partsColl[i];
    }
  }
  return cand;
}

std::vector<const reco::GenParticle*> 
TopGenEvent::lightQuarks(bool includingBQuarks) const 
{
  std::vector<const reco::GenParticle*> lightQuarks;
  for (reco::GenParticleCollection::const_iterator part = parts_->begin(); part < parts_->end(); ++part) {
    if( (includingBQuarks && std::abs(part->pdgId())==TopDecayID::bID) || std::abs(part->pdgId())<TopDecayID::bID ) {
      if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == nullptr){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
      }
      lightQuarks.push_back( part->clone() );
    }
  }  
  return lightQuarks;
}

std::vector<const reco::GenParticle*> 
TopGenEvent::radiatedGluons(int pdgId) const{
  std::vector<const reco::GenParticle*> rads;
  for (reco::GenParticleCollection::const_iterator part = parts_->begin(); part < parts_->end(); ++part) {
    if ( part->mother() && part->mother()->pdgId()==pdgId ){
      if(part->pdgId()==TopDecayID::glueID){
	if( dynamic_cast<const reco::GenParticle*>( &(*part) ) == nullptr){
	  throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
	}
      }
      rads.push_back( part->clone() );
    }
  }  
  return rads;
}
