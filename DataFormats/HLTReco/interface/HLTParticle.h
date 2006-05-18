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
 *  $Date: 2006/04/26 09:27:44 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include <string>
#include <map>

namespace reco
{

  class HLTParticle : public Particle {
    // Currently, we re-use the Particle class from the Candidate Model
    // Later, we may use our own (more compact) particle (base) class

  public:
    HLTParticle(): Particle() { }
    HLTParticle(char q, const LorentzVector& p4): Particle(q, p4) { }

  };

}

#endif
