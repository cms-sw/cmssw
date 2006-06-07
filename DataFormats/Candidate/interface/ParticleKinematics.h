#ifndef Candidate_ParticleKinematics_h
#define Candidate_ParticleKinematics_h
/** \class reco::ParticleKinematics
 *
 * Base class describing a generic reconstructed Particle with 
 * a momentum 4-vector measurement
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ParticleKinematics.h,v 1.1 2006/06/05 13:40:07 llista Exp $
 *
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
 
namespace reco {

  class ParticleKinematics {
  public:
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// default constructor
    ParticleKinematics() { }
    /// constructor from values
    ParticleKinematics( const LorentzVector & p4 ) : 
      p4_( p4 ) { }
    /// destructor
    virtual ~ParticleKinematics() { }
    /// four-momentum Lorentz vector
    const LorentzVector & p4() const { return p4_; }
    /// spatial momentum vector
    Vector momentum() const { return p4_.Vect(); }
    /// boost vector to boost a Lorentz vector 
    /// to the particle center of mass system
    Vector boostToCM() const { return p4_.BoostToCM(); }
    /// magnitude of momentum vector
    double p() const { return p4_.P(); }
    /// energy
    double energy() const { return p4_.E(); }  
    /// transverse energy
    double et() const { return p4_.Et(); }  
    /// mass
    double mass() const { return p4_.M(); }
    /// mass squared
    double massSqr() const { return p4_.M2(); }
    /// transverse mass
    double mt() const { return p4_.Mt(); }
    /// transverse mass squared
    double mtSqr() const { return p4_.Mt2(); }
    /// x coordinate of momentum vector
    double px() const { return p4_.Px(); }
    /// y coordinate of momentum vector
    double py() const { return p4_.Py(); }
    /// z coordinate of momentum vector
    double pz() const { return p4_.Pz(); }
    /// transverse momentum
    double pt() const { return p4_.Pt(); }
    /// momentum azimuthal angle
    double phi() const { return p4_.Phi(); }
    /// momentum polar angle
    double theta() const { return p4_.Theta(); }
    /// momentum pseudorapidity
    double eta() const { return p4_.Eta(); }
    /// repidity
    double y() const { return p4_.Rapidity(); }
  protected:
    /// four-momentum Lorentz vector
    LorentzVector p4_;
  };

}

#endif
