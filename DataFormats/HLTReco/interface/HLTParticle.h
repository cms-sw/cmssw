#ifndef HLTReco_HLTParticle_h
#define HLTReco_HLTParticle_h

/** \class HLTParticle
 *
 *
 *  The basic information (4-momentum etc.) on a reconstrcuted physics
 *  object (e/gamma/mu/jet/Met...) for those physics objects used in
 *  an HLT filter decision.
 *
 *
 *  $Date: 2006/06/17 04:02:16 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Candidate/interface/Particle.h"

namespace reco
{

  class HLTParticle : public Particle {
    // Currently, we extend the Particle class from the Candidate Model with:

  protected:
    char id_; // numerical PDG id of particle ("fundamental" particles only!)

  public:
    HLTParticle(): Particle(), id_() { }

    HLTParticle(const Particle& p, int id=0) : Particle(p), id_() {
      if (id<-100) {id_=-101;} else if (id>+100) {id_=+101;} else {id_=char(id);}
    }

    int id() const { return (int) (id_); }

  };

}

#endif
