#ifndef PhysicsTools_HepMCCandAlgos_GenParticlesHelper_h
#define PhysicsTools_HepMCCandAlgos_GenParticlesHelper_h

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "HepPDT/ParticleID.hh"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"


#include <iostream>

namespace MCTruthHelper {
  
  /////////////////////////////////////////////////////////////////////////////
  //these are robust, generator-independent functions for categorizing
  //mainly final state particles, but also intermediate hadrons
  //or radiating leptons
  
  //is particle prompt (not from hadron, muon, or tau decay)
  template<typename P>
  bool isPrompt(const P &p);  
  
  //is particle prompt and final state  
  template<typename P>
  bool isPromptFinalState(const P &p);

  //is particle prompt and decayed  
  template<typename P>
  bool isPromptDecayed(const P &p);
  
  //this particle is a direct or indirect tau decay product
  template<typename P>
  bool isTauDecayProduct(const P &p);

  //this particle is a direct or indirect decay product of a prompt tau
  template<typename P>
  bool isPromptTauDecayProduct(const P &p);
  
  //this particle is a direct tau decay product
  template<typename P>
  bool isDirectTauDecayProduct(const P &p);

  //this particle is a direct decay product from a prompt tau 
  template<typename P>
  bool isDirectPromptTauDecayProduct(const P &p);

  //this particle is a direct or indirect muon decay product
  template<typename P>
  bool isMuonDecayProduct(const P &p);

  //this particle is a direct or indirect decay product of a prompt muon
  template<typename P>
  bool isPromptMuonDecayProduct(const P &p);    
  
  //this particle is a direct decay product from a hadron
  template<typename P>
  bool isDirectHadronDecayProduct(const P &p);

  //is particle a hadron
  template<typename P>
  bool isHadron(const P &p);
  
  /////////////////////////////////////////////////////////////////////////////
  //these are generator history-dependent functions for tagging particles
  //associated with the hard process
  //Currently implemented for Pythia 6 and Pythia 8 status codes and history
  
  //this particle is part of the hard process
  template<typename P>
  bool isHardProcess(const P &p);  
  
  //this particle is the direct descendant of a hard process particle of the same pdg id
  template<typename P>
  bool fromHardProcess(const P &p);  
  
  //this particle is the final state direct descendant of a hard process particle  
  template<typename P>
  bool fromHardProcessFinalState(const P &p);

  //this particle is the decayed direct descendant of a hard process particle
  //such as a tau from the hard process
  template<typename P>
  bool fromHardProcessDecayed(const P &p);  

  //this particle is a direct or indirect decay product of a tau
  //from the hard process
  template<typename P>
  bool isHardProcessTauDecayProduct(const P &p);  

  //this particle is a direct decay product of a tau
  //from the hard process
  template<typename P>
  bool isDirectHardProcessTauDecayProduct(const P &p);  
  
  //this particle is the last copy of the particle in the chain with the same pdg id
  //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)
  template<typename P>
  bool isLastCopy(const P &p);
  
  /////////////////////////////////////////////////////////////////////////////
  //These are utility functions used by the above
  
  //first mother in chain with a different pdg than the particle
  template<typename P>
  const P *uniqueMother(const P &p);
  
  //return first copy of particle in chain (may be the particle itself)
  template<typename P>
  const P *firstCopy(const P &p);
  
  //return last copy of particle in chain (may be the particle itself)
  template<typename P>
  const P *lastCopy(const P &p);
  
  //return next copy of particle in chain (0 in case this is already the last copy)
  template<typename P>
  const P *nextCopy(const P &p);
  
  //return mother matching requested abs(pdgid) and status
  template<typename P>
  const P *findMother(const P &p, int abspdgid, int status);

  /////////////////////////////////////////////////////////////////////////////
  //These are very basic utility functions to implement a common interface for reco::GenParticle
  //and HepMC::GenParticle  
  
  //pdgid
  int pdgId(const reco::GenParticle &p);

  //pdgid
  int pdgId(const HepMC::GenParticle &p);  
  
  //abs(pdgid)
  int absPdgId(const reco::GenParticle &p);

  //abs(pdgid)
  int absPdgId(const HepMC::GenParticle &p);
  
  //mother
  const reco::GenParticle *mother(const reco::GenParticle &p);

  //mother
  const HepMC::GenParticle *mother(const HepMC::GenParticle &p);

  //number of daughters
  unsigned int numberOfDaughters(const reco::GenParticle &p);  
  
  //number of daughters
  unsigned int numberOfDaughters(const HepMC::GenParticle &p);
  
  //ith daughter
  const reco::GenParticle *daughter(const reco::GenParticle &p, unsigned int idau);  
  
  //ith daughter
  const HepMC::GenParticle *daughter(const HepMC::GenParticle &p, unsigned int idau);  
  
  /////////////////////////////////////////////////////////////////////////////
  //Helper function to fill status flags
  template<typename P>
  void fillGenStatusFlags(const P &p, reco::GenStatusFlags &statusFlags);

}
  
namespace MCTruthHelper {
  
  /////////////////////////////////////////////////////////////////////////////
  //implementations
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPrompt(const P &p) {
    const P *um = uniqueMother(p);
    if (!um) return true;
    
    //particle from hadron/muon/tau decay -> not prompt
    int ampdg = absPdgId(*um);
    if ( um->status() == 2 && (isHadron(*um) || ampdg==13 || ampdg==15) ) {
      return false;
    }
    
    return true;
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPromptFinalState(const P &p) {
    return p.status()==1 && isPrompt(p);
  }
  
  template<typename P>
  bool isPromptDecayed(const P &p) {
    return p.status()==2 && isPrompt(p);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isTauDecayProduct(const P &p) {
    return findMother(p,15,2) != 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPromptTauDecayProduct(const P &p) {
    const P *tau = findMother(p,15,2);
    return tau && isPrompt(*tau);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isDirectTauDecayProduct(const P &p) {
    const P *um = uniqueMother(p);
    return um && absPdgId(*um)==15 && um->status()==2;
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isDirectPromptTauDecayProduct(const P &p) {
    const P *um = uniqueMother(p);
    return um && absPdgId(*um)==15 && um->status()==2 && isPrompt(*um);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isMuonDecayProduct(const P &p) {
    return findMother(p,13,2) != 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPromptMuonDecayProduct(const P &p) {
    const P *mu = findMother(p,13,2);
    return mu && isPrompt(*mu);
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isDirectHadronDecayProduct(const P &p) {
    const P *um = uniqueMother(p);
    return um && isHadron(*um) && um->status()==2;
  }  

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isHadron(const P &p) {
    HepPDT::ParticleID heppdtid(pdgId(p));
    return heppdtid.isHadron();
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isHardProcess(const P &p) {
    
    //status 3 in pythia6 means hard process;
    if (p.status()==3) return true;
    
    //hard process codes for pythia8 are 21-29 inclusive (currently 21,22,23,24 are used)
    if (p.status()>20 && p.status()<30) return true;
    
    //if this is a final state or decayed particle,
    //check if direct mother is a resonance decay in pythia8 but exclude FSR branchings
    //(In pythia8 if a resonance decay product did not undergo any further branchings
    //it will be directly stored as status 1 or 2 without any status 23 copy)
    if (p.status()==1 || p.status()==2) {
      const P *um = mother(p);
      if (um) {
        bool fromResonanceDecay = firstCopy(*um)->status()==22;
        
        const P *umNext = nextCopy(*um);
        bool fsrBranching = umNext && umNext->status()>50 && umNext->status()<60;
        
        if (fromResonanceDecay && !fsrBranching) return true;
      }
    }
    
    return false;
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool fromHardProcess(const P &p) {
    return isHardProcess(*firstCopy(p));
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool fromHardProcessFinalState(const P &p) {
    return p.status()==1 && fromHardProcess(p);
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool fromHardProcessDecayed(const P &p) {
    return p.status()==2 && fromHardProcess(p);
  }
  
  /////////////////////////////////////////////////////////////////////////////  
  template<typename P>
  bool isHardProcessTauDecayProduct(const P &p) {
    const P *tau = findMother(p,15,2);
    return tau && fromHardProcessDecayed(*tau);    
  }

  /////////////////////////////////////////////////////////////////////////////  
  template<typename P>
  bool isDirectHardProcessTauDecayProduct(const P &p) {
    const P *um = uniqueMother(p);
    return um && absPdgId(*um)==15 && um->status()==2 && fromHardProcess(*um);    
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isLastCopy(const P &p) {
    return &p == lastCopy(p);
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *uniqueMother(const P &p) {
    const P *mo = &p;
    while (mo && pdgId(*mo)==pdgId(p)) {
      mo = mother(*mo);
    }
    return mo;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *firstCopy(const P &p) {
    const P *pcopy = &p;
    while (mother(*pcopy) && pdgId(*mother(*pcopy))==pdgId(p)) {
      pcopy = mother(*pcopy);
    }
    return pcopy;    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *lastCopy(const P &p) {
    const P *pcopy = &p;
    bool hasDaughterCopy = true;
    while (hasDaughterCopy) {
      hasDaughterCopy = false;
      const unsigned int ndau = numberOfDaughters(*pcopy);
      for (unsigned int idau = 0; idau<ndau; ++idau) {
        const P *dau = daughter(*pcopy,idau);
        if (pdgId(*dau)==pdgId(p)) {
          pcopy = dau;
          hasDaughterCopy = true;
          break;
        }
      }
    }
    return pcopy;       
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *nextCopy(const P &p) {
    
    const unsigned int ndau = numberOfDaughters(p);
    for (unsigned int idau = 0; idau<ndau; ++idau) {
      const P *dau = daughter(p,idau);
      if (pdgId(*dau)==pdgId(p)) {
        return dau;
      }
    }
    
    return 0;     
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *findMother(const P &p, int abspdgid, int status) {
    const P *mo = mother(p);
    while (mo && (absPdgId(*mo)!=abspdgid || mo->status()!=status) ) {
      mo = mother(*mo);
    }
    return mo;
  }
  
  //////////////////////////////////////////////////////////////
  int pdgId(const reco::GenParticle &p) {
    return p.pdgId();
  }

  //////////////////////////////////////////////////////////////
  int pdgId(const HepMC::GenParticle &p) {
    return p.pdg_id();
  }  
  
  //////////////////////////////////////////////////////////////
  int absPdgId(const reco::GenParticle &p) {
    return std::abs(p.pdgId());
  }

  //////////////////////////////////////////////////////////////
  int absPdgId(const HepMC::GenParticle &p) {
    return std::abs(p.pdg_id());
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *mother(const reco::GenParticle &p) {
    return static_cast<const reco::GenParticle*>(p.mother());
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const HepMC::GenParticle *mother(const HepMC::GenParticle &p) {
    return p.production_vertex() && p.production_vertex()->particles_in_size() ? *p.production_vertex()->particles_in_const_begin() : 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  unsigned int numberOfDaughters(const reco::GenParticle &p) {
    return p.numberOfDaughters();
  }
  
  /////////////////////////////////////////////////////////////////////////////
  unsigned int numberOfDaughters(const HepMC::GenParticle &p) {
    return p.end_vertex() ? p.end_vertex()->particles_out_size() : 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *daughter(const reco::GenParticle &p, unsigned int idau) {
    return static_cast<const reco::GenParticle*>(p.daughter(idau));
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const HepMC::GenParticle *daughter(const HepMC::GenParticle &p, unsigned int idau) {
    return *(p.end_vertex()->particles_out_const_begin() + idau);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  void fillGenStatusFlags(const P &p, reco::GenStatusFlags &statusFlags) {
    statusFlags.setIsPrompt(isPrompt(p));
    statusFlags.setIsTauDecayProduct(isTauDecayProduct(p));
    statusFlags.setIsPromptTauDecayProduct(isPromptTauDecayProduct(p));
    statusFlags.setIsDirectTauDecayProduct(isDirectTauDecayProduct(p));
    statusFlags.setIsDirectPromptTauDecayProduct(isDirectPromptTauDecayProduct(p));
    statusFlags.setIsMuonDecayProduct(isMuonDecayProduct(p));
    statusFlags.setIsPromptMuonDecayProduct(isPromptMuonDecayProduct(p));
    statusFlags.setIsDirectHadronDecayProduct(isDirectHadronDecayProduct(p));
    statusFlags.setIsHardProcess(isHardProcess(p));
    statusFlags.setFromHardProcess(fromHardProcess(p));
    statusFlags.setIsHardProcessTauDecayProduct(isHardProcessTauDecayProduct(p));
    statusFlags.setIsDirectHardProcessTauDecayProduct(isDirectHardProcessTauDecayProduct(p));
    statusFlags.setIsLastCopy(isLastCopy(p));
  }  
  
}

#endif
