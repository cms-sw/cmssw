//
// $Id: Particle.h,v 1.4 2008/06/03 22:28:07 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Particle_h
#define DataFormats_PatCandidates_Particle_h

/**
  \class    pat::Particle Particle.h "DataFormats/PatCandidates/interface/Particle.h"
  \brief    Analysis-level particle class

   Particle implements an analysis-level particle class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Particle.h,v 1.4 2008/06/03 22:28:07 gpetrucc Exp $
*/

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"

// Define typedefs for convenience
namespace pat {
  class Particle;
  typedef std::vector<Particle>              ParticleCollection; 
  typedef edm::Ref<ParticleCollection>       ParticleRef; 
  typedef edm::RefVector<ParticleCollection> ParticleRefVector; 
}

namespace pat {


  typedef reco::LeafCandidate ParticleType;


  class Particle : public PATObject<ParticleType> {

    public:

      Particle();
      Particle(const ParticleType & aParticle);
      virtual ~Particle();

      virtual Particle * clone() const { return new Particle(*this); }

  };


}

#endif
