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

  //is particle a decayed hadron, muon, or tau (does not include resonance decays like W,Z,Higgs,top,etc)
  //This flag is equivalent to status 2 in the current HepMC standard
  //but older generators (pythia6, herwig6) predate this and use status 2 also for other intermediate
  //particles/states
  template<typename P>
  bool isDecayedLeptonHadron(const P &p);  
  
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
  
  //this particle is the direct descendant of a hard process particle of the same pdg id
  //For outgoing particles the kinematics are those before QCD or QED FSR
  //This corresponds roughly to status code 3 in pythia 6
  template<typename P>
  bool fromHardProcessBeforeFSR(const P &p);  
  
  //this particle is the first copy of the particle in the chain with the same pdg id
  template<typename P>
  bool isFirstCopy(const P &p);  
  
  //this particle is the last copy of the particle in the chain with the same pdg id
  //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)
  template<typename P>
  bool isLastCopy(const P &p);
  
  //this particle is the last copy of the particle in the chain with the same pdg id
  //before QED or QCD FSR
  //(and therefore is more likely, but not guaranteed, to carry the momentum after ISR)
  //This flag only really makes sense for outgoing particles
  template<typename P>
  bool isLastCopyBeforeFSR(const P &p);    
  
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

  //return last copy of particle in chain before QED or QCD FSR (may be the particle itself)
  template<typename P>
  const P *lastCopyBeforeFSR(const P &p);
  
  //return last copy of particle in chain before QED or QCD FSR, starting from itself (may be the particle itself)
  template<typename P>
  const P *lastDaughterCopyBeforeFSR(const P &p);  
  
  //return mother copy which is a hard process particle
  template<typename P>
  const P *hardProcessMotherCopy(const P &p);  
  
  //return previous copy of particle in chain (0 in case this is already the first copy)
  template<typename P>
  const P *previousCopy(const P &p);  
  
  //return next copy of particle in chain (0 in case this is already the last copy)
  template<typename P>
  const P *nextCopy(const P &p);

  //return decayed mother (walk up the chain until found)
  template<typename P>
  const P *findDecayedMother(const P &p);
  
  //return decayed mother matching requested abs(pdgid) (walk up the chain until found)
  template<typename P>
  const P *findDecayedMother(const P &p, int abspdgid);

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

  //number of mothers
  unsigned int numberOfMothers(const reco::GenParticle &p);  
  
  //number of mothers
  unsigned int numberOfMothers(const HepMC::GenParticle &p);
  
  //mother
  const reco::GenParticle *mother(const reco::GenParticle &p, unsigned int imoth=0);

  //mother
  const HepMC::GenParticle *mother(const HepMC::GenParticle &p, unsigned int imoth=0);

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
    //particle from hadron/muon/tau decay -> not prompt
    //checking all the way up the chain treats properly the radiated photon
    //case as well
    return findDecayedMother(p) == 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPromptFinalState(const P &p) {
    return p.status()==1 && isPrompt(p);
  }
  
  template<typename P>
  bool isDecayedLeptonHadron(const P &p) {
    return p.status()==2  && (isHadron(p) || absPdgId(p)==13 || absPdgId(p)==15) && isLastCopy(p);
  }
  
  template<typename P>
  bool isPromptDecayed(const P &p) {
    return isDecayedLeptonHadron(p) && isPrompt(p);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isTauDecayProduct(const P &p) {
    return findDecayedMother(p,15) != 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPromptTauDecayProduct(const P &p) {
    const P *tau = findDecayedMother(p,15);
    return tau && isPrompt(*tau);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isDirectTauDecayProduct(const P &p) {
    const P *tau = findDecayedMother(p,15);
    const P *dm = findDecayedMother(p);
    return tau && tau==dm;
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isDirectPromptTauDecayProduct(const P &p) {
    const P *tau = findDecayedMother(p,15);
    const P *dm = findDecayedMother(p);
    return tau && tau==dm && isPrompt(*tau);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isMuonDecayProduct(const P &p) {
    return findDecayedMother(p,13) != 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isPromptMuonDecayProduct(const P &p) {
    const P *mu = findDecayedMother(p,13);
    return mu && isPrompt(*mu);
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isDirectHadronDecayProduct(const P &p) {
    const P *um = uniqueMother(p);
    return um && isHadron(*um) && isDecayedLeptonHadron(*um);
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
        bool fromResonance = firstCopy(*um)->status()==22;
        
        const P *umNext = nextCopy(*um);
        bool fsrBranching = umNext && umNext->status()>50 && umNext->status()<60;
        
        if (fromResonance && !fsrBranching) return true;
      }
    }
    
    return false;
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool fromHardProcess(const P &p) {
    return hardProcessMotherCopy(p) != 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool fromHardProcessFinalState(const P &p) {
    return p.status()==1 && fromHardProcess(p);
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool fromHardProcessDecayed(const P &p) {
    return isDecayedLeptonHadron(p) && fromHardProcess(p);
  }
  
  /////////////////////////////////////////////////////////////////////////////  
  template<typename P>
  bool isHardProcessTauDecayProduct(const P &p) {
    const P *tau = findDecayedMother(p,15);
    return tau && fromHardProcessDecayed(*tau);    
  }

  /////////////////////////////////////////////////////////////////////////////  
  template<typename P>
  bool isDirectHardProcessTauDecayProduct(const P &p) {
    const P *tau = findDecayedMother(p,15);
    const P *dm = findDecayedMother(p);
    return tau && tau==dm && fromHardProcess(*tau);    
  }  
  
  template<typename P>
  bool fromHardProcessBeforeFSR(const P &p) {
    //pythia 6 documentation line roughly corresponds to this condition
    if (p.status()==3) return true;
    
    //check hard process mother properties
    const P *hpc = hardProcessMotherCopy(p);
    if (!hpc) return false;
        
    //for incoming partons in pythia8, more useful information is not
    //easily available, so take only the incoming parton itself
    if (hpc->status()==21 && (&p)==hpc) return true;
    
    //for intermediate particles in pythia 8, just take the last copy
    if (hpc->status()==22 && isLastCopy(p)) return true;
    
    //for outgoing particles in pythia 8, explicitly find the last copy
    //before FSR starting from the hardProcess particle, and take only
    //this one
    if ( (hpc->status()==23 || hpc->status()==1) && (&p)==lastDaughterCopyBeforeFSR(*hpc) ) return true;
    
    
    //didn't satisfy any of the conditions
    return false;
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isFirstCopy(const P &p) {
    return &p == firstCopy(p);
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isLastCopy(const P &p) {
    return &p == lastCopy(p);
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  bool isLastCopyBeforeFSR(const P &p) {
    return &p == lastCopyBeforeFSR(p);
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
    while (previousCopy(*pcopy)) {
      pcopy = previousCopy(*pcopy);
    }
    return pcopy;    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *lastCopy(const P &p) {
    const P *pcopy = &p;
    while (nextCopy(*pcopy)) {
      pcopy = nextCopy(*pcopy);
    }
    return pcopy;    
  }
    
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *lastCopyBeforeFSR(const P &p) {
    //start with first copy and then walk down until there is FSR
    const P *pcopy = firstCopy(p);
    bool hasDaughterCopy = true;
    while (hasDaughterCopy) {
      hasDaughterCopy = false;
      const unsigned int ndau = numberOfDaughters(*pcopy);
      //look for FSR
      for (unsigned int idau = 0; idau<ndau; ++idau) {
        const P *dau = daughter(*pcopy,idau);
        if (dau && (pdgId(*dau)==21 || pdgId(*dau)==22)) {
          //has fsr (or else decayed and is the last copy by construction)
          return pcopy;
        }        
      }
      //look for daughter copy
      for (unsigned int idau = 0; idau<ndau; ++idau) {
        const P *dau = daughter(*pcopy,idau);
        if (dau && (pdgId(*dau)==pdgId(p))) {
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
  const P *lastDaughterCopyBeforeFSR(const P &p) {
    //start with this particle and then walk down until there is FSR
    const P *pcopy = &p;
    bool hasDaughterCopy = true;
    while (hasDaughterCopy) {
      hasDaughterCopy = false;
      const unsigned int ndau = numberOfDaughters(*pcopy);
      //look for FSR
      for (unsigned int idau = 0; idau<ndau; ++idau) {
        const P *dau = daughter(*pcopy,idau);
        if (dau && (pdgId(*dau)==21 || pdgId(*dau)==22)) {
          //has fsr (or else decayed and is the last copy by construction)
          return pcopy;
        }        
      }
      //look for daughter copy
      for (unsigned int idau = 0; idau<ndau; ++idau) {
        const P *dau = daughter(*pcopy,idau);
        if (dau && pdgId(*dau)==pdgId(p)) {
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
  const P *hardProcessMotherCopy(const P &p) {
    //is particle itself is hard process particle
    if (isHardProcess(p)) return &p;
    
    //check if any other copies are hard process particles
    const P *pcopy = &p;
    while (previousCopy(*pcopy)) {
      pcopy = previousCopy(*pcopy);
      if (isHardProcess(*pcopy)) return pcopy;
    }
    return 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *previousCopy(const P &p) {
    
    const unsigned int nmoth = numberOfMothers(p);
    for (unsigned int imoth = 0; imoth<nmoth; ++imoth) {
      const P *moth = mother(p,imoth);
      if (moth && (pdgId(*moth)==pdgId(p))) {
        return moth;
      }
    }
    
    return 0;     
  }   
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *nextCopy(const P &p) {
    
    const unsigned int ndau = numberOfDaughters(p);
    for (unsigned int idau = 0; idau<ndau; ++idau) {
      const P *dau = daughter(p,idau);
      if (dau && (pdgId(*dau)==pdgId(p))) {
        return dau;
      }
    }
    
    return 0;     
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *findDecayedMother(const P &p) {
    const P *mo = mother(p);
    while (mo && !isDecayedLeptonHadron(*mo)) {
      mo = mother(*mo);
    }
    return mo;
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  const P *findDecayedMother(const P &p, int abspdgid) {
    const P *mo = mother(p);
    while (mo && (absPdgId(*mo)!=abspdgid || !isDecayedLeptonHadron(*mo)) ) {
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
  unsigned int numberOfMothers(const reco::GenParticle &p) {
    return p.numberOfMothers();
  }
  
  /////////////////////////////////////////////////////////////////////////////
  unsigned int numberOfMothers(const HepMC::GenParticle &p) {
    return p.production_vertex() ? p.production_vertex()->particles_in_size() : 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *mother(const reco::GenParticle &p, unsigned int imoth) {
    return static_cast<const reco::GenParticle*>(p.mother(imoth));
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const HepMC::GenParticle *mother(const HepMC::GenParticle &p, unsigned int imoth) {
    if (p.production_vertex() && p.production_vertex()->particles_in_size()){
         HepMC::GenParticle *mother_cand=*(p.production_vertex()->particles_in_const_begin() + imoth);
         //Sherpa Fix to prevent circular relations between mother and daughter
         if (mother_cand && (p.status()==1 || (mother_cand->end_vertex()->id()< p.end_vertex()->id()))){
            return mother_cand;
         } 
    }
    return 0;
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
  
  const HepMC::GenParticle *daughter(const HepMC::GenParticle &p, unsigned int idau) {
    HepMC::GenParticle *daughter_cand = *(p.end_vertex()->particles_out_const_begin() + idau);
    //Sherpa Fix to prevent circular relations between mother and daughter
    if (daughter_cand->status()==1 || (daughter_cand->end_vertex()->id() > p.end_vertex()->id()))
      return daughter_cand;
    else 
      return 0;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  template<typename P>
  void fillGenStatusFlags(const P &p, reco::GenStatusFlags &statusFlags) {
    statusFlags.setIsPrompt(isPrompt(p));
    statusFlags.setIsDecayedLeptonHadron(isDecayedLeptonHadron(p));
    statusFlags.setIsTauDecayProduct(isTauDecayProduct(p));
    statusFlags.setIsPromptTauDecayProduct(isPromptTauDecayProduct(p));
    statusFlags.setIsDirectTauDecayProduct(isDirectTauDecayProduct(p));
    statusFlags.setIsDirectPromptTauDecayProduct(isDirectPromptTauDecayProduct(p));
    statusFlags.setIsDirectHadronDecayProduct(isDirectHadronDecayProduct(p));
    statusFlags.setIsHardProcess(isHardProcess(p));
    statusFlags.setFromHardProcess(fromHardProcess(p));
    statusFlags.setIsHardProcessTauDecayProduct(isHardProcessTauDecayProduct(p));
    statusFlags.setIsDirectHardProcessTauDecayProduct(isDirectHardProcessTauDecayProduct(p));
    statusFlags.setFromHardProcessBeforeFSR(fromHardProcessBeforeFSR(p));
    statusFlags.setIsFirstCopy(isFirstCopy(p));
    statusFlags.setIsLastCopy(isLastCopy(p));
    statusFlags.setIsLastCopyBeforeFSR(isLastCopyBeforeFSR(p));
  }  
  
}

#endif
