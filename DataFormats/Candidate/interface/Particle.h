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
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
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
    Particle();
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true );
    /// constructor from values
    Particle( Charge q, const PolarLorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ),
	      int pdgId = 0, int status = 0, bool integerCharge = true );
    void swap(Particle& other);
    /// destructor
    virtual ~Particle();
    // copy ctor
    Particle(const Particle& srv);
    // assignment operator
    Particle& operator=(const Particle& rhs);
    /// electric charge
    int charge() const;
    /// set electric charge
    void setCharge( Charge q );
    /// electric charge
    int threeCharge() const;
    /// set electric charge
    void setThreeCharge( Charge qx3 );
    /// four-momentum Lorentz vector
    const LorentzVector & p4() const;
    /// four-momentum Lorentz vector
    const PolarLorentzVector & polarP4() const;
    /// spatial momentum vector
    Vector momentum() const;
    /// boost vector to boost a Lorentz vector 
    /// to the particle center of mass system
    Vector boostToCM() const;
    /// magnitude of momentum vector
    double p() const;
    /// energy
    double energy() const;
    /// transverse energy 
    double et() const;
    /// mass
    double mass() const;
    /// mass squared
    double massSqr() const;
    /// transverse mass
    double mt() const;
    /// transverse mass squared
    double mtSqr() const;
    /// x coordinate of momentum vector
    double px() const;
    /// y coordinate of momentum vector
    double py() const;
    /// z coordinate of momentum vector
    double pz() const;
    /// transverse momentum
    double pt() const;
    /// momentum azimuthal angle
    double phi() const;
    /// momentum polar angle
    double theta() const;
    /// momentum pseudorapidity
    double eta() const;
    /// repidity
    double rapidity() const;
    /// repidity
    double y() const;
    /// set 4-momentum
    void setP4( const LorentzVector & p4 );
    /// set 4-momentum
    void setP4( const PolarLorentzVector & p4 );
    /// set particle mass
    void setMass( double m );
    /// set Pz
    void setPz( double pz );
    /// vertex position
    const Point & vertex() const;
    /// x coordinate of vertex position
    double vx() const;
    /// y coordinate of vertex position
    double vy() const;
    /// z coordinate of vertex position
    double vz() const;
    /// set vertex
    void setVertex( const Point & vertex );
    /// PDG identifier
    int pdgId() const;
    // set PDG identifier
    void setPdgId( int pdgId );
    /// status word
    int status() const;
    /// set status word
    void setStatus( int status );
    /// long lived flag
    static const unsigned int longLivedTag;
    /// set long lived flag
    void setLongLived();
    /// is long lived?
    bool longLived() const;

  protected:
    /// electric charge
    Charge qx3_;   
    /// four-momentum Lorentz vector
    float pt_, eta_, phi_, mass_;
    /// vertex position
    Point vertex_;
    /// PDG identifier
    int pdgId_;
    /// status word
    int status_;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    /// internal cache for p4
    mutable std::atomic<PolarLorentzVector*> p4Polar_;
    /// internal cache for p4
    mutable std::atomic<LorentzVector*> p4Cartesian_;
#else
    /// internal cache for p4
    mutable PolarLorentzVector* p4Polar_;
    /// internal cache for p4
    mutable LorentzVector* p4Cartesian_;
#endif
    /// set internal cache
    void cachePolar() const;
    /// set internal cache
    void cacheCartesian() const;
    /// clear internal cache
    void clearCache() const;
  };

}

#endif
