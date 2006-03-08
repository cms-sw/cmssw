#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
 
namespace reco {

  class Particle {
  public:
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// electric charge type
    typedef char Charge;
    /// default constructor
    Particle() { }
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      p4_( p4 ), vtx_( vtx ), q_( q ) { }
    /// destructor
    virtual ~Particle() { }
    /// electric charge
    int charge() const { return q_; }
    /// four-momentum Lorentz vector
    const LorentzVector & p4() const { return p4_; }
    /// spatial momentum vector
    Vector p3() const { return p4_.Vect(); }
    /// boost vector to boost a Lorentz vector 
    /// to the particle center of mass system
    Vector boostToCM() const { return p4_.BoostToCM(); }
    /// magnitude of momentum vector
    double momentum() const { return p4_.P(); }
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
    /// momentum magnitude ( same as momentum() )
    double p() const { return momentum(); }
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
    /// vertex position
    const Point & vertex() const { return vtx_; }
    /// x coordinate of vertex position
    double x() const { return vtx_.X(); }
    /// y coordinate of vertex position
    double y() const { return vtx_.Y(); }
    /// z coordinate of vertex position
    double z() const { return vtx_.Z(); }
  protected:
    /// four-momentum Lorentz vector
    LorentzVector p4_;
    /// vertex position
    Point vtx_;
    /// electric charge
    Charge q_;    
  };

}

#endif
