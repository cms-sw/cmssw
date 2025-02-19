//
// $Id: Particle.h,v 1.6 2008/11/28 19:02:15 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Particle_h
#define DataFormats_PatCandidates_Particle_h

/**
  \class    pat::Particle Particle.h "DataFormats/PatCandidates/interface/Particle.h"
  \brief    Analysis-level particle class

   Particle implements an analysis-level particle class within the 'pat'
   namespace.

  \author   Steven Lowette, Giovanni Petrucciani
  \version  $Id: Particle.h,v 1.6 2008/11/28 19:02:15 lowette Exp $
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


  class Particle : public PATObject<reco::LeafCandidate> {

    public:

      /// default constructor
      Particle();
      /// constructor from a LeafCandidate
      Particle(const reco::LeafCandidate & aParticle);
      /// destructor
      virtual ~Particle();

      /// required reimplementation of the Candidate's clone method
      virtual Particle * clone() const { return new Particle(*this); }

  };


}

#endif
