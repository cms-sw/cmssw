#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

std::vector<const reco::Candidate*>
TtSemiLepEvtPartons::vec(const TtGenEvent& genEvt)
{
  std::vector<const reco::Candidate*> vec;
  if(genEvt.isSemiLeptonic()) {
    vec.resize(4);
    vec[LightQ   ] = genEvt.hadronicDecayQuark()    ? genEvt.hadronicDecayQuark()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[LightQBar] = genEvt.hadronicDecayQuarkBar() ? genEvt.hadronicDecayQuarkBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[HadB     ] = genEvt.hadronicDecayB()        ? genEvt.hadronicDecayB()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[LepB     ] = genEvt.leptonicDecayB()        ? genEvt.leptonicDecayB()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
  }
  else {
    for(unsigned i=0; i<4; i++)
      vec.push_back( new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
  }
  return vec;
}
