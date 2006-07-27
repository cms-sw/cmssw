#ifndef HepMCCandidate_HepMCCandidate_h
#define HepMCCandidate_HepMCCandidate_h
/** \class reco::HepMCCandidate
 *
 * particle candidate from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: HepMCCandidate.h,v 1.5 2006/04/07 11:38:36 llista Exp $
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace HepMC {
  class GenParticle;
}

namespace reco {

  class HepMCCandidate : public LeafCandidate {
  public:
    /// reference to HepMC::GenParticle
    typedef const HepMC::GenParticle * GenParticleRef;
    /// default constructor
    HepMCCandidate() : LeafCandidate(), genParticle_( 0 ) { }
    /// constroctor from pointer to generator particle
    HepMCCandidate( const HepMC::GenParticle * );
    /// destructor
    virtual ~HepMCCandidate();
    /// pointer to generator particle
    GenParticleRef genParticle() const { return genParticle_; }
    /// return a clone
    HepMCCandidate * clone() const;

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// pointer to generator particle
    GenParticleRef genParticle_;
  };

  /// get GenParticle component
  GET_DEFAULT_CANDIDATE_COMPONENT( HepMCCandidate, HepMCCandidate::GenParticleRef, genParticle );
}

#endif
