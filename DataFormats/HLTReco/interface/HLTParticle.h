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
 *  $Date: 2006/06/17 21:40:25 $
 *  $Revision: 1.6 $
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
      if (id<-110) {
	id_=-110;
      } else if (id>+110) {
	id_=+110;
      } else {
	id_=static_cast<char>(id);
      }
      // hence up to |id|<110 free to use, subject to PDG assigned values!
    }

    int id() const { return static_cast<int>(id_); }

  };

}

#endif
