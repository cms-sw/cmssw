#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Particle.h,v 1.9 2006/10/10 09:10:50 llista Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
 
namespace reco {

  class Particle {
  public:
    /// electric charge type
    typedef char Charge;
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// point in the space
    typedef math::XYZVector Vector;
    /// default constructor
    Particle() { }
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ) ) : 
       q_( q ), p4_( p4 ), vertex_( vertex ) { }
    /// destructor
    virtual ~Particle() { }
    /// electric charge
    int charge() const { return q_; }
    /// set electric charge
    void setCharge( Charge q ) { q_ = q; }
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
    double rapidity() const { return p4_.Rapidity(); }
    /// repidity
    double y() const { return p4_.Rapidity(); }
    /// set 4-momentum
    void setP4( const LorentzVector & p4 ) { p4_ = p4; }
    /// vertex position
    const Point & vertex() const { return vertex_; }
    /// x coordinate of vertex position
    double vx() const { return vertex_.X(); }
    /// y coordinate of vertex position
    double vy() const { return vertex_.Y(); }
    /// z coordinate of vertex position
    double vz() const { return vertex_.Z(); }
    /// set vertex
    void setVertex( const Point & vertex ) { vertex_ = vertex; }
  protected:
    /// electric charge
    Charge q_;   
    /// four-momentum Lorentz vector
    LorentzVector p4_;
    /// vertex position
    Point vertex_;
  };

}

#endif
