#ifndef HepMCCandidate_HepMCCandidate_h
#define HepMCCandidate_HepMCCandidate_h
/** \class reco::HepMCCandidate
 *
 * particle candidate from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: HepMCCandidate.h,v 1.3 2006/03/08 12:57:08 llista Exp $
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

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// pointer to generator particle
    GenParticleRef genParticle_;
  };

  /// get GenParticle component
  GET_CANDIDATE_COMPONENT( HepMCCandidate, HepMCCandidate::GenParticleRef, DefaultComponentTag, genParticle );
}

#endif
