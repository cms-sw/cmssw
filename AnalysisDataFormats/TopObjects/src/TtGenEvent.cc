//
// $Id: TtGenEvent.cc,v 1.35 2012/07/03 13:11:27 davidlt Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

/// default constructor from decaySubset and initSubset
TtGenEvent::TtGenEvent(reco::GenParticleRefProd& decaySubset, reco::GenParticleRefProd& initSubset)
{
  parts_ = decaySubset;
  initPartons_= initSubset;
  if(top() && topBar())
    topPair_ = math::XYZTLorentzVector(top()->p4()+topBar()->p4());
}

bool
TtGenEvent::fromGluonFusion() const
{
  const reco::GenParticleCollection& initPartsColl = *initPartons_;
  if(initPartsColl.size()==2)
    if(initPartsColl[0].pdgId()==21 && initPartsColl[1].pdgId()==21)
      return true;
  return false;
}

bool
TtGenEvent::fromQuarkAnnihilation() const
{
  const reco::GenParticleCollection& initPartsColl = *initPartons_;
  if(initPartsColl.size()==2)
    if(std::abs(initPartsColl[0].pdgId())<TopDecayID::tID && initPartsColl[0].pdgId()==-initPartsColl[1].pdgId())
      return true;
  return false;
}

WDecay::LeptonType 
TtGenEvent::semiLeptonicChannel() const 
{
  WDecay::LeptonType type=WDecay::kNone;
  if( isSemiLeptonic() && singleLepton() ){
    if( std::abs(singleLepton()->pdgId())==TopDecayID::elecID ) type=WDecay::kElec;
    if( std::abs(singleLepton()->pdgId())==TopDecayID::muonID ) type=WDecay::kMuon;
    if( std::abs(singleLepton()->pdgId())==TopDecayID::tauID  ) type=WDecay::kTau;
  }
  return type;
}

std::pair<WDecay::LeptonType, WDecay::LeptonType>
TtGenEvent::fullLeptonicChannel() const 
{
  WDecay::LeptonType typeA=WDecay::kNone, typeB=WDecay::kNone;  
  if( isFullLeptonic() ){
    if( lepton() ){
      if( std::abs(lepton()->pdgId())==TopDecayID::elecID ) typeA=WDecay::kElec;
      if( std::abs(lepton()->pdgId())==TopDecayID::muonID ) typeA=WDecay::kMuon;
      if( std::abs(lepton()->pdgId())==TopDecayID::tauID  ) typeA=WDecay::kTau;      
    }
    if( leptonBar() ){
      if( std::abs(leptonBar()->pdgId())==TopDecayID::elecID ) typeB=WDecay::kElec;
      if( std::abs(leptonBar()->pdgId())==TopDecayID::muonID ) typeB=WDecay::kMuon;
      if( std::abs(leptonBar()->pdgId())==TopDecayID::tauID  ) typeB=WDecay::kTau;      
    }
  }
  return ( std::pair<WDecay::LeptonType,WDecay::LeptonType>(typeA, typeB) );
}

const reco::GenParticle* 
TtGenEvent::lepton(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand = 0;
  const reco::GenParticleCollection& partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isLepton(partsColl[i]) && partsColl[i].mother() &&
	std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])>0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonBar(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand = 0;
  const reco::GenParticleCollection& partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isLepton(partsColl[i]) && partsColl[i].mother() &&
	std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])<0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::singleLepton(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand = 0;
  if( isSemiLeptonic(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isLepton(partsColl[i]) && partsColl[i].mother() &&
	  std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrino(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isNeutrino(partsColl[i]) && partsColl[i].mother() &&
	std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])>0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::neutrinoBar(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  const reco::GenParticleCollection & partsColl = *parts_;
  for (unsigned int i = 0; i < partsColl.size(); ++i) {
    if (reco::isNeutrino(partsColl[i]) && partsColl[i].mother() &&
	std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
      if(reco::flavour(partsColl[i])<0){
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::singleNeutrino(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( isSemiLeptonic(excludeTauLeptons) ) {
    const reco::GenParticleCollection & partsColl = *parts_;
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (reco::isNeutrino(partsColl[i]) && partsColl[i].mother() &&
	  std::abs(partsColl[i].mother()->pdgId())==TopDecayID::WID) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayQuark(bool invertFlavor) const 
{
  const reco::GenParticle* cand=0;
  // catch W boson and check its daughters for a quark; 
  // make sure the decay is semi-leptonic first; this 
  // only makes sense if taus are not excluded from the 
  // decision
  if( singleLepton(false) ){
    for(reco::GenParticleCollection::const_iterator w=parts_->begin(); w!=parts_->end(); ++w){
      if( std::abs( w->pdgId() )==TopDecayID::WID ){
	// make sure that the particle is a W daughter
	for(reco::GenParticle::const_iterator wd=w->begin(); wd!=w->end(); ++wd){ 
	  // make sure that the parton is a quark
	  if( std::abs(wd->pdgId())<TopDecayID::tID ){
	    if( invertFlavor?reco::flavour(*wd)<0:reco::flavour(*wd)>0 ){
	      cand = dynamic_cast<const reco::GenParticle* > (&(*wd));
	      if(cand == 0){
		throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
	      }
	      break;
	    }
	  }
	}
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayB(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (std::abs(partsColl[i].pdgId())==TopDecayID::bID && 
	  reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayW(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (std::abs(partsColl[i].pdgId())==TopDecayID::WID && 
          reco::flavour(singleLep) != -reco::flavour(partsColl[i])) { 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
	cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::hadronicDecayTop(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (std::abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)==reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayB(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (std::abs(partsColl[i].pdgId())==TopDecayID::bID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i])) {
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayW(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for (unsigned int i = 0; i < partsColl.size(); ++i) {
      if (std::abs(partsColl[i].pdgId())==TopDecayID::WID &&
          reco::flavour(singleLep) == - reco::flavour(partsColl[i])) { 
	// PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* 
TtGenEvent::leptonicDecayTop(bool excludeTauLeptons) const 
{
  const reco::GenParticle* cand=0;
  if( singleLepton(excludeTauLeptons) ){
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton(excludeTauLeptons);
    for( unsigned int i = 0; i < partsColl.size(); ++i ){
      if( std::abs(partsColl[i].pdgId())==TopDecayID::tID &&
          reco::flavour(singleLep)!=reco::flavour(partsColl[i]) ){
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

std::vector<const reco::GenParticle*> TtGenEvent::leptonicDecayTopRadiation(bool excludeTauLeptons) const{
  if( leptonicDecayTop(excludeTauLeptons) ){
    return (leptonicDecayTop(excludeTauLeptons)->pdgId()>0 ? radiatedGluons(TopDecayID::tID) : radiatedGluons(-TopDecayID::tID));
  }
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}

std::vector<const reco::GenParticle*> TtGenEvent::hadronicDecayTopRadiation(bool excludeTauLeptons) const{
  if( hadronicDecayTop(excludeTauLeptons) ){
    return (hadronicDecayTop(excludeTauLeptons)->pdgId()>0 ? radiatedGluons(TopDecayID::tID) : radiatedGluons(-TopDecayID::tID));
  }
  std::vector<const reco::GenParticle*> rad;
  return (rad);
}
