#ifndef CandUtils_Thrust_h
#define CandUtils_Thrust_h
// $Id: Thrust.h,v 1.7 2006/01/31 11:50:46 llista Exp $
//
// Ported from original BaBar implementation
//
// Luca Lista
//
#include "DataFormats/Math/interface/Vector3D.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

namespace reco {
  class Candidate;
}
 
class Thrust  {
public:
  typedef math::XYZVector Vector;
  typedef reco::CandidateCollection::const_iterator const_iterator;
  Thrust( const_iterator begin, const_iterator end );
  double thrust() const { return thrust_; } 
  const Vector& axis() const { return axis_; } 

private:
  double thrust_;
  Vector axis_;
  double pSum_;
  const unsigned int n_;
  std::vector<Vector> p_;

  struct ThetaPhi {
    ThetaPhi( double t, double p ) : theta( t ), phi( p ) { }
    double theta, phi;
  };
  double thrust(const Vector & theAxis) const; 
  ThetaPhi initialAxis() const;
  ThetaPhi finalAxis( ThetaPhi ) const;
  Vector axis( double theta, double phi ) const;
  Vector axis( const ThetaPhi & tp ) const  {
    return axis( tp.theta, tp.phi );
  }
  void parabola( double & a, double & b, double & c, 
		 const Vector &, const Vector &, const Vector & ) const;
};

#endif
