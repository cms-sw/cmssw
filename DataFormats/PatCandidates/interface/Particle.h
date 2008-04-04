//
// $Id: Particle.h,v 1.2 2008/03/05 14:47:33 fronga Exp $
//

#ifndef DataFormats_PatCandidates_Particle_h
#define DataFormats_PatCandidates_Particle_h

/**
  \class    pat::Particle Particle.h "DataFormats/PatCandidates/interface/Particle.h"
  \brief    Analysis-level particle class

   Particle implements an analysis-level particle class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Particle.h,v 1.2 2008/03/05 14:47:33 fronga Exp $
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

      virtual Particle * clone() const { return new Particle(*this); }

  };


}

#endif
