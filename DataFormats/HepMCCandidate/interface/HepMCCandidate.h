#ifndef HepMCCandidate_HepMCCandidate_h
#define HepMCCandidate_HepMCCandidate_h
/** class reco::HepMCCandidate
 *
 * particle candidate from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: HepMCCandidate.h,v 1.1 2006/03/08 09:19:50 llista Exp $
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

  /// get GenParticle component 
  template<>
  struct component<HepMC::GenParticle> {
    static const HepMC::GenParticle * get( const Candidate & c ) {
      const HepMCCandidate * dc = dynamic_cast<const HepMCCandidate *>( & c );
      if ( dc == 0 ) return 0;
      return dc->genParticle();
    }
  };

}

#endif
