#ifndef PhysicsTools_HepMCCandAlgos_GenParticlesHelper_h
#define PhysicsTools_HepMCCandAlgos_GenParticlesHelper_h

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"

#include <iostream>

namespace MCTruthHelper {
  
  /////////////////////////////////////////////////////////////////////////////
  //these are robust, generator-independent functions for categorizing
  //mainly final state particles, but also intermediate hadrons
  //or radiating leptons
  
  //is particle prompt (not from hadron, muon, or tau decay
  bool isPrompt(const reco::GenParticle &p);  
  
  //is particle prompt and final state  
  bool isPromptFinalState(const reco::GenParticle &p);  
  
  //this particle is a direct or indirect tau decay product
  bool isTauDecayProduct(const reco::GenParticle &p);

  //this particle is a direct or indirect decay product of a prompt tau
  bool isPromptTauDecayProduct(const reco::GenParticle &p);
  
  //this particle is a direct tau (or muon) decay product
  bool isDirectTauDecayProduct(const reco::GenParticle &p);

  //this particle is a direct decay product from a prompt tau 
  bool isDirectPromptTauDecayProduct(const reco::GenParticle &p);

  //this particle is a direct or indirect muon decay product
  bool isMuonDecayProduct(const reco::GenParticle &p);

  //this particle is a direct or indirect decay product of a prompt muon
  bool isPromptMuonDecayProduct(const reco::GenParticle &p);    
  
  //this particle is a direct decay product from a hadron
  bool isDirectHadronDecayProduct(const reco::GenParticle &p);

  //is particle a hadron
  bool isHadron(const reco::GenParticle &p);
  
  /////////////////////////////////////////////////////////////////////////////
  //these are generator history-dependent functions for tagging particles
  //associated with the hard process
  //Currently implemented for Pythia 6 and Pythia 8 status codes and history
  
  //this particle is part of the hard process
  bool isHardProcess(const reco::GenParticle &p);  
  
  //this particle is the direct descendant of a hard process particle
  bool fromHardProcess(const reco::GenParticle &p);  
  
  //this particle is the final state direct descendant of a hard process particle  
  bool fromHardProcessFinalState(const reco::GenParticle &p);

  //this particle is the last copy of the particle in the chain
  //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)
  bool isLastCopy(const reco::GenParticle &p);
  
  /////////////////////////////////////////////////////////////////////////////
  //These are utility functions used by the above
  
  //first mother in chain with a different pdg than the particle
  const reco::GenParticle *uniqueMother(const reco::GenParticle &p);
  
  //return first copy of particle in chain (may be the particle itself)
  const reco::GenParticle *firstCopy(const reco::GenParticle &p);
  
  //return last copy of particle in chain (may be the particle itself)
  const reco::GenParticle *lastCopy(const reco::GenParticle &p);
  
  //return next copy of particle in chain (0 in case this is already the last copy)
  const reco::GenParticle *nextCopy(const reco::GenParticle &p);
  
  //return mother matching requested abs(pdgid) and status
  const reco::GenParticle *findMother(const reco::GenParticle &p, int abspdgid, int status);
  
  /////////////////////////////////////////////////////////////////////////////
  //Helper function to fill status flags
  void fillGenStatusFlags(const reco::GenParticle &p, reco::GenStatusFlags &statusFlags);

}

#endif
