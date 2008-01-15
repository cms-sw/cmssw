//
// $Id: Particle.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Particle_h
#define DataFormats_PatCandidates_Particle_h

/**
  \class    Particle Particle.h "DataFormats/PatCandidates/interface/Particle.h"
  \brief    Analysis-level particle class

   Particle implements an analysis-level particle class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Particle.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  typedef reco::Particle ParticleType;


  class Particle : public PATObject<ParticleType> {

    public:

      Particle();
      Particle(const ParticleType & aParticle);
      virtual ~Particle();

  };


}

#endif
