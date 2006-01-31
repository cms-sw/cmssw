#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <cmath>
using namespace aod;
typedef math::XYZTLorentzVector LorentzVector;

const double pi = M_PI, pi2 = 2 * pi, pi_2 = pi / 2, pi_4 = pi / 4;
const double epsilon = 0.0001;
const int nSegsTheta = 10; // number of initial segments in theta
const int nSegsPhi = 10; // number of initial segments in phi
const int nSegs = nSegsTheta * nSegsPhi; // total number of segments

Thrust::Thrust( const_iterator begin, const_iterator end ) : 
  thrust_( 0 ), axis_( 0, 0, 0 ), pSum_( 0 ), 
  n_( end - begin ), p_( n_ ) {
  if ( n_ == 0 ) return;
  int i = 0;
  for( const_iterator t = begin; t != end; ++ t, ++ i )
    pSum_ += ( p_[ i ] = t->p3() ).r();

  axis_ = axis( finalAxis( initialAxis() ) );
  if ( cos( axis_.theta() ) < 0 ) axis_ *= -1;
  thrust_ = thrust( axis_ );
}

Thrust::ThetaPhi Thrust::initialAxis() const {
  int i, j;
  double thr[ nSegs ], max = 0;
  int indI = 0, indJ = 0, index = -1;
  for ( i = 0; i < nSegsTheta ; ++i ) {
    double z = cos( pi * i / ( nSegsTheta - 1 ) );
    double r = sqrt( 1 - z * z );
    for ( j = 0; j < nSegsPhi ; ++j ) {
      double phi = pi2 * j / nSegsPhi;
      thr[ i * nSegsPhi + j ] = thrust( Vector( r * cos( phi ), r * sin( phi ), z ) );
      if ( thr[ i * nSegsPhi + j ] > max ) {
        index = i * nSegsPhi + j;
        indI = i; indJ = j;
        max = thr[ index ];
      }
    }
  }

  // take max and one point on either size, fitting to a parabola and
  // extrapolating to the real max.  Do this separately for each dimension.
  // y = ax^2 + bx + c.  At the max, x = 0, on either side, x = +/-1.
  // do phi first
  double a, b, c = max;
  int ind1 = indJ + 1;
  if ( ind1 >= nSegsPhi ) ind1 -= nSegsPhi;
  int ind2 = indJ - 1;
  if ( ind2 < 0 ) ind2 += nSegsPhi;
  a = ( thr[ ind1 ] + thr[ ind2 ] - 2 * c ) / 2;
  b = thr[ ind1 ] - a - c;
  double maxPhiInd = 0;
  if ( a != 0 ) maxPhiInd = - b / ( 2 * a );
  double maxThetaInd;
  if ( indI == 0 || indI == ( nSegsTheta - 1 ) ) 
    maxThetaInd = indI;
  else {
    ind1 = indI + 1;
    ind2 = indI - 1;
    a = ( thr[ ind1 ] + thr[ ind2 ] - 2 * c ) / 2;
    b = thr[ind1] - a - c; 
    maxThetaInd = 0;
    if ( a != 0 ) maxThetaInd = - b / ( 2 * a );
  }

  return ThetaPhi( pi * ( maxThetaInd + indI ) / ( nSegsTheta - 1 ),
		   pi2 * ( maxPhiInd + indJ ) / nSegsPhi );
}

Thrust::ThetaPhi Thrust::finalAxis( ThetaPhi best ) const {
  double maxChange1, maxChange2, a, b, c, thr;
  int mandCt = 3, maxCt = 1000;
  bool done;
  do { 
    Vector axis1 = axis( best );
    Vector axis2 = axis( best.theta + epsilon, best.phi );
    Vector axis3 = axis( best.theta - epsilon, best.phi );
    c = thrust( axis1 );
    a = ( thrust( axis2 ) - 2 * c + thrust( axis3 ) ) / 2;
    b = thrust( axis2 ) - a - c; 
    maxChange1 = 10 * ( b < 0 ? -1 : 1 );
    if ( a != 0 ) maxChange1 = -b / ( 2 * a );
    while ( fabs( maxChange1 * epsilon ) > pi_4 ) maxChange1 /= 2;
    if ( maxChange1 == 0 && ( best.theta == 0 || best.theta == pi ) ) { 
      best.phi += pi_2;
      if ( best.phi > pi2 ) best.phi -= pi2;
      axis1 = axis( best );
      axis2 = axis( best.theta + epsilon, best.phi );
      axis3 = axis( best.theta - epsilon, best.phi );
      double t2 = thrust( axis2 );
      c = thrust( axis1 );
      a = ( t2 - 2 * c + thrust( axis3 ) ) / 2;
      b = t2 - a - c;
      maxChange1 = 10 * ( b < 0 ? -1 : 1 ); // linear 
      if ( a != 0 ) maxChange1 = - b / ( 2 * a );
    }
    do {
      axis2 = axis( best.theta + maxChange1 * epsilon, best.phi );
      // L.L.: fixed odd behavoir adding epsilon (???)
      thr = thrust( axis2 ) + epsilon;
      if ( thr < c ) maxChange1 /= 2;
    } while ( thr < c );

    best.theta += maxChange1 * epsilon;
    if ( best.theta > pi) {
      best.theta = pi - ( best.theta - pi );
      best.phi += pi;
      if ( best.phi > pi2 ) best.phi -= pi2;
    }
    if ( best.theta < 0 ) {
      best.theta *= -1;
      best.phi += pi;
      if ( best.phi > pi2 ) best.phi -= pi2;
    }
    axis1 = axis( best );
    axis2 = axis( best.theta, best.phi + epsilon );
    axis3 = axis( best.theta, best.phi - epsilon );
    double t2 = thrust( axis2 );
    c = thrust( axis1 );
    a = ( t2 - 2 * c + thrust( axis3 ) ) / 2;
    b = t2 - a - c;
    maxChange2 = 10 * ( b < 0 ? -1 : 1 );
    if ( a != 0 ) maxChange2 = - b / ( 2 * a );
    while ( fabs( maxChange2 * epsilon ) > pi_4 ) { maxChange2 /= 2; }
    do {
      axis2 = axis( best.theta, best.phi + maxChange2 * epsilon );
      // L.L.: fixed odd behavoir adding epsilon
      thr = thrust( axis2 ) + epsilon;
      if ( thr < c ) maxChange2 /= 2;
    } while ( thr < c );
    best.phi += maxChange2 * epsilon;
    if ( best.phi > pi2 ) best.phi -= pi2;
    if ( best.phi < 0 ) best.phi += pi2;
    if ( mandCt > 0 ) mandCt--;
    maxCt--;
    done = ( maxChange1 * maxChange1 > 1 ||
	     maxChange2 * maxChange2 > 1 ||
	     mandCt ) && ( maxCt > 0 );
  } while ( done );

  return best;
}

Thrust::Vector Thrust::axis( double theta, double phi ) const {
  double theSin = sin( theta );
  return Vector( theSin * cos(phi), theSin * sin(phi), cos(theta) );
}

double Thrust::thrust( const Vector & axis ) const {
  double result = 0;
  double sum = 0;
  for ( unsigned int i = 0; i < n_; ++i )
    sum += fabs( axis.Dot( p_[i] ) );
  if ( pSum_ > 0 ) result = sum / pSum_;
  return result;
}
