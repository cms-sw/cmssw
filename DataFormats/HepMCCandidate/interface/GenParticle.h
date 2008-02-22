#ifndef HepMCCandidate_GenParticle_h
#define HepMCCandidate_GenParticle_h
/** \class reco::GenParticle
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticle.h,v 1.3 2007/11/05 15:43:54 llista Exp $
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
    GenParticle(const Particle & p) : 
      CompositeRefCandidateT<GenParticleRefVector>(p) { }
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

  private:
    /// checp overlap with another candidate
    bool overlap(const Candidate &) const;
 };

}

#endif
