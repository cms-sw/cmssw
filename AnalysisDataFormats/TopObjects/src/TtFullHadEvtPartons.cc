#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

std::vector<const reco::Candidate*>
TtFullHadEvtPartons::vec(const TtGenEvent& genEvt)
{
  std::vector<const reco::Candidate*> vec;
  if(genEvt.isFullHadronic()) {
    vec[LightQTop      ] = genEvt.lightQFromTop()       ? genEvt.lightQFromTop()       : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[LightQBarTop   ] = genEvt.lightQBarFromTop()    ? genEvt.lightQBarFromTop()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[B              ] = genEvt.b()                   ? genEvt.b()                   : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[LightQTopBar   ] = genEvt.lightQFromTopBar()    ? genEvt.lightQFromTopBar()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[LightQBarTopBar] = genEvt.lightQBarFromTopBar() ? genEvt.lightQBarFromTopBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[BBar           ] = genEvt.bBar()                ? genEvt.bBar()                : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
  }
  else {
    for(unsigned i=0; i<6; i++)
      vec.push_back( new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
  }
  return vec;
}
