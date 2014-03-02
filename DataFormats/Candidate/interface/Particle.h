#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 *
 */

#include "ParticleState.h"
namespace reco {
  
  class Particle : public ParticleState {
  public:

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    using ParticleState::ParticleState;
#else
    // for Reflex to parse...  (compilation will use the above)
    Particle();
    Particle( Charge q, const PtEtaPhiMass  & p4, const Point & vertex= Point( 0, 0, 0 ),
	      int pdgId=0, int status=0, bool integerCharge=true);
    Particle( Charge q, const LorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true );
    Particle( Charge q, const PolarLorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true );
#endif
    /// destructor
    virtual ~Particle(){}

    void construct(int qx3,  float pt, float eta, float phi, float mass, const Point & vtx, int pdgId, int status) {
      ParticleState::operator=(ParticleState(qx3, PolarLorentzVector(pt,eta,phi,mass), vtx, pdgId, status, false));
    }
    

  };

}

#endif
