#ifndef CandUtils_Thrust_h
#define CandUtils_Thrust_h
// $Id: Thrust.h,v 1.1 2006/01/31 08:24:14 llista Exp $
// Ported from BaBar implementation
#include "DataFormats/Math/interface/Vector3D.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

namespace aod {
  class Candidate;
}
 
class Thrust  {
public:
  typedef math::XYZVector Vector;
  typedef aod::CandidateCollection::const_iterator const_iterator;
  Thrust( const_iterator begin, const_iterator end );
  double thrust() const { return _thrust; } 
  const Vector& axis() const { return _axis; } 

private:

  // Member data 
  double _thrust;
  Vector _axis;
  double _pSum;
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
};

#endif
