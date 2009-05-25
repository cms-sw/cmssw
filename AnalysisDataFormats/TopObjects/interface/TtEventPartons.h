#ifndef TtEventPartons_h
#define TtEventPartons_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include <vector>

// ----------------------------------------------------------------------
// common wrapper class to fill partons in a well
// defined order for ttbar event in the:
//
//  * full leptonic
//  * semi-leptonic
//  * full hadronic
//
// decay channel used for jet parton matching in 
// in the TtJetPartonMatch template in TopTools; 
// the order is defined in TtGenEvent
//
// ----------------------------------------------------------------------


class TtFullLepEventPartons {

 public:

  /// empty constructor
  TtFullLepEventPartons(){};
  /// default destructor
  ~TtFullLepEventPartons(){};

  /// return vector of partons in the order defined in TtGenEvent
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( (genEvt.isFullLeptonic() && genEvt.b()   ) ? genEvt.b()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullLeptonic() && genEvt.bBar()) ? genEvt.bBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    return vec;
  };
};


class TtSemiLepEventPartons {

 public:

  /// empty constructor
  TtSemiLepEventPartons(){};
  /// default destructor
  ~TtSemiLepEventPartons(){};

  /// return vector of partons in the order defined in TtGenEvent
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.hadronicDecayQuark()   ) ? genEvt.hadronicDecayQuark()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.hadronicDecayQuarkBar()) ? genEvt.hadronicDecayQuarkBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false));
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.hadronicDecayB()       ) ? genEvt.hadronicDecayB()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.leptonicDecayB()       ) ? genEvt.leptonicDecayB()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    return vec;
  };
};


class TtFullHadEventPartons {

 public:

  /// empty constructor
  TtFullHadEventPartons(){};
  /// default destructor
  ~TtFullHadEventPartons(){};

  /// return vector of partons in the order defined in TtGenEvent
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( (genEvt.isFullHadronic() && genEvt.lightQFromTop()      ) ? genEvt.lightQFromTop()       : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.lightQBarFromTop()   ) ? genEvt.lightQBarFromTop()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.b()                  ) ? genEvt.b()                   : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.lightQFromTopBar()   ) ? genEvt.lightQFromTopBar()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.lightQBarFromTopBar()) ? genEvt.lightQBarFromTopBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.bBar()               ) ? genEvt.bBar()                : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    return vec;
  };
};


#endif
