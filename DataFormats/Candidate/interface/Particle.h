#ifndef Candidate_Particle_h
#define Candidate_Particle_h
// $Id: Particle.h,v 1.13 2006/02/21 10:37:32 llista Exp $
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
 
namespace reco {

  class Particle {
  public:
    typedef math::XYZTLorentzVector LorentzVector;
    typedef math::XYZVector Vector;
    typedef math::XYZPoint Point;
    typedef char Charge;
    Particle() { }
    Particle( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      p4_( p4 ), vtx_( vtx ), q_( q ) { }
    virtual ~Particle() { }
    int charge() const { return q_; }
    const LorentzVector & p4() const { return p4_; }
    Vector p3() const { return p4_.Vect(); }
    Vector boostToCM() const { return p4_.BoostToCM(); }
    double momentum() const { return p4_.P(); }
    double energy() const { return p4_.E(); }  
    double et() const { return p4_.Et(); }  
    double mass() const { return p4_.M(); }
    double massSqr() const { return p4_.M2(); }
    double mt() const { return p4_.Mt(); }
    double mtSqr() const { return p4_.Mt2(); }
    double p() const { return momentum(); }
    double px() const { return p4_.Px(); }
    double py() const { return p4_.Py(); }
    double pz() const { return p4_.Pz(); }
    double pt() const { return p4_.Pt(); }
    double phi() const { return p4_.Phi(); }
    double theta() const { return p4_.Theta(); }
    double eta() const { return p4_.Eta(); }
    const Point & vertex() const { return vtx_; }
    double x() const { return vtx_.X(); }
    double y() const { return vtx_.Y(); }
    double z() const { return vtx_.Z(); }
  protected:
    LorentzVector p4_;
    Point vtx_;
    Charge q_;    
  };

}

#endif
