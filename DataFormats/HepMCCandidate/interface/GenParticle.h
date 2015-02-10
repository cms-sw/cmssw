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
    
  private:
    /// checp overlap with another candidate
    bool overlap(const Candidate &) const;
    int collisionId_;
    GenStatusFlags statusFlags_;
 };

}

#endif
