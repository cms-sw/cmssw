#ifndef CandUtils_Thrust_h
#define CandUtils_Thrust_h
// $Id: Thrust.h,v 1.1 2006/01/31 08:24:14 llista Exp $
// Ported from BaBar implementation
#include "DataFormats/Math/interface/Vector3D.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include <vector>

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
  double _denom_sum;
  const unsigned int n_;

  void calc_denom(const std::vector<double> & X, const std::vector<double> & Y,
		  const std::vector<double> & Z);

  double calc_thrust(const std::vector<double> & theAxis, const std::vector<double> & X,
		     const std::vector<double> & Y, const std::vector<double> & Z) const; 

  std::vector<double> get_initial_axis(const std::vector<double> & X, const std::vector<double> & Y,
			    const std::vector<double> & Z) const;

  std::vector<double> get_final_axis(double thetaInit, double phiInit, const std::vector<double> &  X,
			  const std::vector<double> & Y, const std::vector<double> & Z) const;

  std::vector<double> get_axis(double theta,double phi) const; // take care of memory
};

#endif
