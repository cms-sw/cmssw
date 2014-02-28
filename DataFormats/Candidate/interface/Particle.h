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

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Rtypes.h"

namespace reco {
  
  class Particle {
  public:
    /// electric charge type
    typedef int Charge;
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// Lorentz vector
    typedef math::PtEtaPhiMLorentzVector PolarLorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// point in the space
    typedef math::XYZVector Vector;
    /// default constructor
    Particle() : vertex_(0, 0, 0), pt_(0), eta_(0), phi_(0), mass_(0),
		 qx3_(0), pdgId_(0), status_(0){}
    
    /// constructor from values
    Particle( Charge q, const PtEtaPhiMass  & p4, const Point & vertex= Point( 0, 0, 0 ),
	      int pdgId=0, int status=0, bool integerCharge=true)
      : vertex_( vertex ),  pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
	p4Polar_(pt_, eta_, phi_, mass_), p4Cartesian_(p4Polar_),
	qx3_( integerCharge ? q*3 : q ),pdgId_( pdgId ), status_( status ){}
    
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true ) :
      vertex_( vertex ),  pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
      p4Polar_(p4), p4Cartesian_(p4),
      qx3_( integerCharge ? q*3 : q ),pdgId_( pdgId ), status_( status ){}

    
    /// constructor from values
    Particle( Charge q, const PolarLorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true ):
      vertex_( vertex ),  pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
      p4Polar_(p4), p4Cartesian_(p4),
      qx3_( integerCharge ? q*3 : q ),pdgId_( pdgId ), status_( status ){}
    

    /// set internal cache
    inline void setPolar()  { 
      p4Polar_ = PolarLorentzVector( pt_, eta_, phi_, mass_ );
    }
    /// set internal cache
    inline void setCartesian()  { 
      setPolar();
      p4Cartesian_ = p4Polar_;
    }

    /// destructor
    virtual ~Particle(){}
 
    /// electric charge
    int charge() const { return qx3_ / 3; }
    /// set electric charge
    void setCharge( Charge q ) { qx3_ = q * 3; }
    /// electric charge
    int threeCharge() const { return qx3_; }
    /// set electric charge
    void setThreeCharge( Charge qx3 ) { qx3_ = qx3; }
    /// four-momentum Lorentz vector
    const LorentzVector & p4() const {  return p4Cartesian_; }
    /// four-momentum Lorentz vector
    const PolarLorentzVector & polarP4() const { return p4Polar_; }
    /// spatial momentum vector
    Vector momentum() const {  return p4Cartesian_.Vect(); }
    /// boost vector to boost a Lorentz vector 
    /// to the particle center of mass system
    Vector boostToCM() const {  return p4Cartesian_.BoostToCM(); }
    /// magnitude of momentum vector
    double p() const {  return p4Cartesian_.P(); }
    /// energy
    double energy() const {  return p4Cartesian_.E(); }  
    /// transverse energy 
    double et() const { return p4Polar_.Et(); }  
    /// mass
    double mass() const { return mass_; }
    /// mass squared
    double massSqr() const { return mass_ * mass_; }
    /// transverse mass
    double mt() const { return p4Polar_.Mt(); }
    /// transverse mass squared
    double mtSqr() const { return p4Polar_.Mt2(); }
    /// x coordinate of momentum vector
    double px() const {  return p4Cartesian_.Px(); }
    /// y coordinate of momentum vector
    double py() const {  return p4Cartesian_.Py(); }
    /// z coordinate of momentum vector
    double pz() const {  return p4Cartesian_.Pz(); }
    /// transverse momentum
    double pt() const { return pt_; }
    /// momentum azimuthal angle
    double phi() const { return phi_; }
    /// momentum polar angle
    double theta() const {  return p4Cartesian_.Theta(); }
    /// momentum pseudorapidity
    double eta() const { return eta_; }
    /// repidity
    double rapidity() const {  return p4Polar_.Rapidity(); }
    /// repidity
    double y() const { return rapidity(); }
    /// set 4-momentum
    void setP4( const LorentzVector & p4 ) { 
      p4Cartesian_ = p4;
      p4Polar_ = p4;
      pt_ = p4Polar_.pt();
      eta_ = p4Polar_.eta();
      phi_ = p4Polar_.phi();
      mass_ = p4Polar_.mass();
    }
    /// set 4-momentum
    void setP4( const PolarLorentzVector & p4 ) { 
      p4Polar_ = p4;
      p4Cartesian_ = p4;
      pt_ = p4Polar_.pt();
      eta_ = p4Polar_.eta();
      phi_ = p4Polar_.phi();
      mass_ = p4Polar_.mass();
    }
    /// set particle mass
    void setMass( double m ) { 
      mass_ = m;
      setCartesian();
      
    }
    void setPz( double pz ) {
      p4Cartesian_.SetPz(pz);
      p4Polar_ = p4Cartesian_;
      pt_ = p4Polar_.pt();
      eta_ = p4Polar_.eta();
      phi_ = p4Polar_.phi();
      mass_ = p4Polar_.mass();
    }
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
    /// set long lived flag
    void setLongLived() { status_ |= longLivedTag; }
    /// is long lived?
    bool longLived() const { return status_ & longLivedTag; }

  private:
    static const unsigned int longLivedTag = 65536;
    
  protected:
    /// vertex position
    Point vertex_;
    
    /// four-momentum Lorentz vector
    float pt_, eta_, phi_, mass_;
    /// internal cache for p4
    PolarLorentzVector p4Polar_;
    /// internal cache for p4
    LorentzVector p4Cartesian_;

    /// electric charge
    Charge qx3_;   

    /// PDG identifier
    int pdgId_;
    /// status word
    int status_;
    
  };

}

#endif
