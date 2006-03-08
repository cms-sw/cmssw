#ifndef HepMCCandidate_HepMCCandidate_h
#define HepMCCandidate_HepMCCandidate_h
/** class reco::HepMCCandidate
 *
 * particle candidate from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: HepMCCandidate.h,v 1.1 2006/02/28 10:59:15 llista Exp $
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace HepMC {
  class GenParticle;
}

namespace reco {

  class HepMCCandidate : public LeafCandidate {
  public:
    /// default constructor
    HepMCCandidate() : LeafCandidate(), genParticle_( 0 ) { }
    /// constroctor from pointer to generator particle
    HepMCCandidate( const HepMC::GenParticle * );
    /// destructor
    virtual ~HepMCCandidate();
    /// pointer to generator particle
    const HepMC::GenParticle * genParticle() const { return genParticle_; }

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// pointer to generator particle
    const HepMC::GenParticle * genParticle_;
  };

}

#endif
