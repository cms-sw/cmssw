#ifndef CandUtils_Thrust_h
#define CandUtils_Thrust_h
// $Id$
// Ported from BaBar implementation
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/own_vector.h"
#include <vector>

namespace aod {
  class Candidate;
}
 
class Thrust  {
public:
  typedef math::XYZVector Vector;
  typedef own_vector<aod::Candidate>::const_iterator const_iterator;
  Thrust();
  void compute( const_iterator begin, const_iterator end );
  void reset();
  double thrust() const { return _thrust; } 
  const Vector& axis() const { return _axis; } 

private:

  // Member data 
  double     _thrust;
  Vector _axis;
  double     _denom_sum;
  bool _cutInCms, _charged;

  //
  // These are original functions by Scott
  //
  void init(const std::vector<double> & valX, const std::vector<double> & valY, 
	    const std::vector<double> & valZ,
	    const double denominator, const unsigned nTracks );

  void calc_denom(const std::vector<double> & X, const std::vector<double> & Y,
		  const std::vector<double> & Z, const unsigned nTracks);

  double calc_thrust(const std::vector<double> & theAxis, const std::vector<double> & X,
		     const std::vector<double> & Y, const std::vector<double> & Z,
		     const unsigned nTracks) const; 

  std::vector<double> get_initial_axis(const std::vector<double> & X, const std::vector<double> & Y,
			    const std::vector<double> & Z, const unsigned nTracks) const;

  std::vector<double> get_final_axis(double thetaInit, double phiInit, const std::vector<double> &  X,
			  const std::vector<double> & Y, const std::vector<double> & Z,
			  const unsigned nTracks) const;

  std::vector<double> get_axis(double theta,double phi) const; // take care of memory
};

#endif
