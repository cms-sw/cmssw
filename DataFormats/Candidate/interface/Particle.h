#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Particle.h,v 1.13 2007/03/05 08:56:51 llista Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
 
namespace reco {
  
  class Particle {
  public:
    /// electric charge type
    typedef int Charge;
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// point in the space
    typedef math::XYZVector Vector;
    /// default constructor
    Particle() : hasCacheSet_( false ) { }
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true ) : 
      qx3_( q ), p4_( p4 ), vertex_( vertex ), pdgId_( pdgId ), status_( status ),
      hasCacheSet_( false ) { 
      if ( integerCharge ) qx3_ *= 3;
    }
    /// destructor
    virtual ~Particle() { }
    /// electric charge
    int charge() const { return qx3_ / 3; }
    /// set electric charge
    void setCharge( Charge q ) { qx3_ = q * 3; }
    /// electric charge
    int threeCharge() const { return qx3_; }
    /// set electric charge
    void setThreeCharge( Charge qx3 ) { qx3_ = qx3; }
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
    double mass() const { cache(); return p4Cache_.M(); }
    /// mass squared
    double massSqr() const { cache(); return p4Cache_.M2(); }
    /// transverse mass
    double mt() const { cache(); return p4Cache_.Mt(); }
    /// transverse mass squared
    double mtSqr() const { cache(); return p4Cache_.Mt2(); }
    /// x coordinate of momentum vector
    double px() const { return p4_.Px(); }
    /// y coordinate of momentum vector
    double py() const { return p4_.Py(); }
    /// z coordinate of momentum vector
    double pz() const { return p4_.Pz(); }
    /// transverse momentum
    double pt() const { cache(); return p4Cache_.Pt(); }
    /// momentum azimuthal angle
    double phi() const { cache(); return p4Cache_.Phi(); }
    /// momentum polar angle
    double theta() const { return p4_.Theta(); }
    /// momentum pseudorapidity
    double eta() const { cache(); return p4Cache_.Eta(); }
    /// repidity
    double rapidity() const { cache(); return p4Cache_.Rapidity(); }
    /// repidity
    double y() const { cache(); return p4Cache_.Rapidity(); }
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
    /// PDG identifier
    int pdgId() const { return pdgId_; }
    // set PDG identifier
    void setPdgId( int pdgId ) { pdgId_ = pdgId; }
    /// status word
    int status() const { return status_; }
    /// set status word
    void setStatus( int status ) { status_ = status; }

  protected:
    /// electric charge
    Charge qx3_;   
    /// four-momentum Lorentz vector
    LorentzVector p4_;
    /// vertex position
    Point vertex_;
    /// PDG identifier
    int pdgId_;
    /// status word
    int status_;
    /// THE FOLLOWING SHOULD BE CHANGED IN 1.7.0
    /// internal cache type for polar coordinates
    typedef math::PtEtaPhiMLorentzVector LorentzVectorCache;
    /// internal cache for p4
    mutable LorentzVectorCache p4Cache_;
    /// has cache been set?
    mutable bool hasCacheSet_;
    /// set internal cache
    void cache() const { 
      if ( hasCacheSet_ ) return;
      p4Cache_ = p4_;
      hasCacheSet_ = true;
    }
  };

}

#endif
