#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Particle.h,v 1.3 2006/05/02 16:13:33 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/ParticleBaseWithCharge.h"
#include "DataFormats/Math/interface/Point3D.h"
 
namespace reco {

  class Particle : public ParticleBaseWithCharge {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// default constructor
    Particle() { }
    /// constructor from values
    Particle( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      ParticleBaseWithCharge( q, p4 ), vtx_( vtx ) { }
    /// destructor
    virtual ~Particle() { }
    /// vertex position
    const Point & vertex() const { return vtx_; }
    /// x coordinate of vertex position
    double x() const { return vtx_.X(); }
    /// y coordinate of vertex position
    double y() const { return vtx_.Y(); }
    /// z coordinate of vertex position
    double z() const { return vtx_.Z(); }
  protected:
    /// vertex position
    Point vtx_;
  };

}

#endif
