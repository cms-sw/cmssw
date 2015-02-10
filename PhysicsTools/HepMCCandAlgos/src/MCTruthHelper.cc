#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "HepPDT/ParticleID.hh"

namespace MCTruthHelper {

  /////////////////////////////////////////////////////////////////////////////
  bool isPrompt(const reco::GenParticle &p) {
    const reco::GenParticle *um = uniqueMother(p);
    if (!um) return true;
    
    //particle from hadron/muon/tau decay -> not prompt
    int ampdg = std::abs(um->pdgId());
    if ( um->status() == 2 && (isHadron(*um) || ampdg==13 || ampdg==15) ) {
      return false;
    }
    
    return true;
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool isPromptFinalState(const reco::GenParticle &p) {
    return p.status()==1 && isPrompt(p);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool isTauDecayProduct(const reco::GenParticle &p) {
    return findMother(p,15,2) != 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool isPromptTauDecayProduct(const reco::GenParticle &p) {
    const reco::GenParticle *tau = findMother(p,15,2);
    return tau && isPrompt(*tau);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool isDirectTauDecayProduct(const reco::GenParticle &p) {
    const reco::GenParticle *um = uniqueMother(p);
    return um && std::abs(um->pdgId())==15 && um->status()==2;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool isDirectPromptTauDecayProduct(const reco::GenParticle &p) {
    const reco::GenParticle *um = uniqueMother(p);
    return um && std::abs(um->pdgId())==15 && um->status()==2 && isPrompt(*um);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool isMuonDecayProduct(const reco::GenParticle &p) {
    return findMother(p,13,2) != 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool isPromptMuonDecayProduct(const reco::GenParticle &p) {
    const reco::GenParticle *mu = findMother(p,13,2);
    return mu && isPrompt(*mu);
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  bool isDirectHadronDecayProduct(const reco::GenParticle &p) {
    const reco::GenParticle *um = uniqueMother(p);
    return um && isHadron(*um) && um->status()==2;
  }  

  /////////////////////////////////////////////////////////////////////////////
  bool isHadron(const reco::GenParticle &p) {
    HepPDT::ParticleID heppdtid(p.pdgId());
    return heppdtid.isHadron();
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool isHardProcess(const reco::GenParticle &p) {
    
    //status 3 in pythia6 means hard process;
    if (p.status()==3) return true;
    
    //hard process codes for pythia8 are 21-29 inclusive (currently 21,22,23,24 are used)
    if (p.status()>20 && p.status()<30) return true;
    
    //check if mother is a resonance decay in pythia8 but exclude FSR branchings
    const reco::GenParticle *um = uniqueMother(p);
    if (um) {
      bool fromResonanceDecay = firstCopy(*um)->status()==22;
      
      const reco::GenParticle *umNext = nextCopy(*um);
      bool fsrBranching = umNext && umNext->status()>50 && umNext->status()<60;
      
      if (fromResonanceDecay && !fsrBranching) return true;
    }
    
    return false;
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool fromHardProcess(const reco::GenParticle &p) {
    return isHardProcess(*firstCopy(p));
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool fromHardProcessFinalState(const reco::GenParticle &p) {
    return p.status()==1 && fromHardProcess(p);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  bool isLastCopy(const reco::GenParticle &p) {
    return &p == lastCopy(p);
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *uniqueMother(const reco::GenParticle &p) {
    const reco::GenParticle *mother = &p;
    while (mother && mother->pdgId()==p.pdgId()) {
      mother = static_cast<const reco::GenParticle*>(mother->mother());
    }
    return mother;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *firstCopy(const reco::GenParticle &p) {
    const reco::GenParticle *pcopy = &p;
    while (pcopy->mother() && pcopy->mother()->pdgId()==p.pdgId()) {
      pcopy = static_cast<const reco::GenParticle*>(pcopy->mother());
    }
    return pcopy;    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *lastCopy(const reco::GenParticle &p) {
    const reco::GenParticle *pcopy = &p;
    bool hasDaughterCopy = true;
    while (hasDaughterCopy) {
      hasDaughterCopy = false;
      for (unsigned int idau = 0; idau<pcopy->numberOfDaughters(); ++idau) {
        const reco::GenParticle *dau = static_cast<const reco::GenParticle*>(pcopy->daughter(idau));
        if (dau->pdgId()==p.pdgId()) {
          pcopy = dau;
          hasDaughterCopy = true;
          break;
        }
      }
    }
    return pcopy;       
  }
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *nextCopy(const reco::GenParticle &p) {
    
    for (unsigned int idau = 0; idau<p.numberOfDaughters(); ++idau) {
      const reco::GenParticle *dau = static_cast<const reco::GenParticle*>(p.daughter(idau));
      if (dau->pdgId()==p.pdgId()) {
        return dau;
      }
    }
    
    return 0;     
  }  
  
  /////////////////////////////////////////////////////////////////////////////
  const reco::GenParticle *findMother(const reco::GenParticle &p, int abspdgid, int status) {
    const reco::GenParticle *mother = static_cast<const reco::GenParticle*>(p.mother());
    while (mother && (std::abs(mother->pdgId())!=abspdgid || mother->status()!=status) ) {
      mother = static_cast<const reco::GenParticle*>(mother->mother());
    }
    return mother;
  }
  
  void fillGenStatusFlags(const reco::GenParticle &p, reco::GenStatusFlags &statusFlags) {
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
    statusFlags.setIsLastCopy(isLastCopy(p));
  }
  
}
