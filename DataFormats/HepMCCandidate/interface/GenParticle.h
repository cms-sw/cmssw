#ifndef HepMCCandidate_GenParticle_h
#define HepMCCandidate_GenParticle_h
/** \class reco::GenParticle
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/CompositeRefCandidateT.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include <vector>

namespace HepMC {
  class GenParticle;
}

namespace reco {

  class GenParticle : public CompositeRefCandidateT<GenParticleRefVector> {
  public:
    /// default constructor
    GenParticle() { }
    /// default constructor
    GenParticle(const LeafCandidate & c) : 
      CompositeRefCandidateT<GenParticleRefVector>(c) { }
    /// constrocturo from values
    GenParticle(Charge q, const LorentzVector & p4, const Point & vtx, 
		int pdgId, int status, bool integerCharge);
    /// constrocturo from values
    GenParticle(Charge q, const PolarLorentzVector & p4, const Point & vtx, 
		int pdgId, int status, bool integerCharge);
    /// destructor
    virtual ~GenParticle();
    /// return a clone
    GenParticle * clone() const;
    void setCollisionId(int s) {collisionId_ = s;}
    int collisionId() const {return collisionId_;}

    const GenStatusFlags &statusFlags() const { return statusFlags_; }
    GenStatusFlags &statusFlags() { return statusFlags_; }
    
    /////////////////////////////////////////////////////////////////////////////
    //basic set of gen status flags accessible directly here
    //the rest accessible through statusFlags()
    //(see GenStatusFlags.h for their meaning)
    
    /////////////////////////////////////////////////////////////////////////////
    //these are robust, generator-independent functions for categorizing
    //mainly final state particles, but also intermediate hadrons/taus
    
    //is particle prompt (not from hadron, muon, or tau decay) and final state
    bool isPromptFinalState() const { return status()==1 && statusFlags_.isPrompt(); }
    
    //is particle prompt (not from hadron, muon, or tau decay) and decayed
    //such as a prompt tau
    bool isPromptDecayed() const { return statusFlags_.isDecayedLeptonHadron() && statusFlags_.isPrompt(); }
    
    //this particle is a direct decay product of a prompt tau and is final state
    //(eg an electron or muon from a leptonic decay of a prompt tau)
    bool isDirectPromptTauDecayProductFinalState() const { return status()==1 && statusFlags_.isDirectPromptTauDecayProduct(); }
    
    /////////////////////////////////////////////////////////////////////////////
    //these are generator history-dependent functions for tagging particles
    //associated with the hard process
    //Currently implemented for Pythia 6 and Pythia 8 status codes and history   
    //and may not have 100% consistent meaning across all types of processes
    //Users are strongly encouraged to stick to the more robust flags above,
    //as well as the expanded set available in GenStatusFlags.h
    
    //this particle is part of the hard process
    bool isHardProcess() const { return statusFlags_.isHardProcess(); }
    
    //this particle is the final state direct descendant of a hard process particle  
    bool fromHardProcessFinalState() const { return status()==1 && statusFlags_.fromHardProcess(); }
    
    //this particle is the decayed direct descendant of a hard process particle
    //such as a tau from the hard process    
    bool fromHardProcessDecayed()    const { return statusFlags_.isDecayedLeptonHadron() && statusFlags_.fromHardProcess(); }
    
    //this particle is a direct decay product of a hardprocess tau and is final state
    //(eg an electron or muon from a leptonic decay of a tau from the hard process)
    bool isDirectHardProcessTauDecayProductFinalState() const { return status()==1 && statusFlags_.isDirectHardProcessTauDecayProduct(); }
    
    //this particle is the direct descendant of a hard process particle of the same pdg id.
    //For outgoing particles the kinematics are those before QCD or QED FSR
    //This corresponds roughly to status code 3 in pythia 6
    //This is the most complex and error prone of all the flags and you are strongly encouraged
    //to consider using the others to fill your needs.
    bool fromHardProcessBeforeFSR()  const { return statusFlags_.fromHardProcessBeforeFSR(); }
    
    //provided for convenience.  Use this one if you were using status 3 before and didn't know or care what it exactly meant
    bool isMostlyLikePythia6Status3() { return fromHardProcessBeforeFSR(); }
    
    //this particle is the last copy of the particle in the chain  with the same pdg id
    //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)    
    bool isLastCopy() const { return statusFlags_.isLastCopy(); }
    
    //this particle is the last copy of the particle in the chain with the same pdg id
    //before QED or QCD FSR
    //(and therefore is more likely, but not guaranteed, to carry the momentum after ISR)  
    bool isLastCopyBeforeFSR() const { return statusFlags_.isLastCopyBeforeFSR(); }
    
  private:
    /// checp overlap with another candidate
    bool overlap(const Candidate &) const;
    int collisionId_;
    GenStatusFlags statusFlags_;
 };

}

#endif
