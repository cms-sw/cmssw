#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullLepEvtPartons.h"

std::vector<const reco::Candidate*>
TtFullLepEvtPartons::vec(const TtGenEvent& genEvt)
{
  std::vector<const reco::Candidate*> vec;
  if(genEvt.isFullLeptonic()) {
    vec.reserve(2);
    vec[B   ] = genEvt.b()    ? genEvt.b()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
    vec[BBar] = genEvt.bBar() ? genEvt.bBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
  }
  else {
    for(unsigned i=0; i<2; i++)
      vec.push_back( new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
  }
  return vec;
}
