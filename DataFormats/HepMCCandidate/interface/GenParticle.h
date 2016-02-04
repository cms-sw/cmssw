#ifndef HepMCCandidate_GenParticle_h
#define HepMCCandidate_GenParticle_h
/** \class reco::GenParticle
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticle.h,v 1.6 2009/09/10 09:50:28 srappocc Exp $
 */
#include "DataFormats/Candidate/interface/CompositeRefCandidateT.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
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

  private:
    /// checp overlap with another candidate
    bool overlap(const Candidate &) const;
    int collisionId_;
 };

}

#endif
