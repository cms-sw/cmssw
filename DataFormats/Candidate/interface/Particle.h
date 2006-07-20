#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Particle.h,v 1.6 2006/06/20 11:27:05 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/ParticleWithCharge.h"
#include "DataFormats/Math/interface/Point3D.h"
 
namespace reco {

  class Particle : public ParticleWithCharge {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// default constructor
    Particle() { }
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vertex = Point( 0, 0, 0 ) ) : 
      ParticleWithCharge( q, p4 ), vertex_( vertex ) { }
    /// destructor
    virtual ~Particle() { }
    /// vertex position
    const Point & vertex() const { return vertex_; }
    /// x coordinate of vertex position
    double vx() const { return vertex_.X(); }
    /// y coordinate of vertex position
    double vy() const { return vertex_.Y(); }
    /// z coordinate of vertex position
    double vz() const { return vertex_.Z(); }
  protected:
    /// vertex position
    Point vertex_;
  };

}

#endif
