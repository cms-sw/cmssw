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
    double invZ = sqrt( 1 - z * z );
    for ( j = 0; j < nSegsPhi ; ++j ) {
      Vector rInitial( invZ * cos( pi2 * j / nSegsPhi ),
		       invZ * sin( pi2 * j / nSegsPhi ),
		       z );
      thr[ i * nSegsPhi + j ] = thrust( rInitial );
      if ( thr[ i * nSegsPhi + j ] > max ) {
        index = i * nSegsPhi + j;
        indI = i;
	indJ = j;
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
  double maxChange1, maxChange2;
  double a,b,c;
  double theThrust;
  int mand_ct = 3; // mandatory number of passes
  int max_ct = 1000; // a very large number of iterations

  int done;
  do { 
    Vector theAxis = axis( best );
    Vector Axis2 = axis( best.theta + epsilon, best.phi ); // do differential
    Vector Axis3 = axis( best.theta - epsilon, best.phi ); // do differential

    // use parabolic approx as above
    c = thrust( theAxis );
    a = ( thrust( Axis2 ) - 2 * c + thrust( Axis3 ) ) / 2;
    b = thrust( Axis2 ) - a - c; 

    maxChange1 = 10 * ( b < 0 ? -1 : 1 ); // linear 
    if ( a != 0 ) maxChange1 = -b / ( 2 * a );

    // make sure change is small to avoid convergence problems
    while ( fabs( maxChange1 * epsilon ) > pi_4 ) maxChange1 /= 2; // small changes

    // special case, use a different phi
    if ( maxChange1 == 0 && ( best.theta == 0 || best.theta == pi ) ) { 
      best.phi += pi_2;
      if ( best.phi > pi2 ) best.phi -= pi2;

      theAxis = axis( best );
      Axis2 = axis( best.theta + epsilon, best.phi ); // do differential
      Axis3 = axis( best.theta - epsilon, best.phi ); // do differential

      // use parabolic approx as above
      double t2 = thrust( Axis2 );
      c = thrust( theAxis );
      a = ( t2 - 2 * c + thrust( Axis3 ) ) / 2;
      b = t2 - a - c;

      maxChange1 = 10 * ( b < 0 ? -1 : 1 ); // linear 
      if ( a != 0 ) maxChange1 = - b / ( 2 * a );
    }

    do {
      Axis2 = axis( best.theta + maxChange1 * epsilon, best.phi );
      // fixed odd behavoir L.L. adding epsilon (???)
      theThrust = thrust( Axis2 ) + epsilon;
      if ( theThrust < c ) maxChange1 /= 2;
    } while ( theThrust < c );

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

    theAxis = axis( best );
    Axis2 = axis( best.theta, best.phi + epsilon ); // do differential
    Axis3 = axis( best.theta, best.phi - epsilon ); // do differential

    // use parabolic approx as above
    double t2 = thrust( Axis2 );
    c = thrust( theAxis );
    a = ( t2 - 2 * c + thrust( Axis3 ) ) / 2;
    b = t2 - a - c;

    maxChange2 = 10 * ( b<0 ? -1 : 1 ); // linear 
    if ( a != 0 ) maxChange2 = - b / ( 2 * a );

    while ( fabs( maxChange2 * epsilon ) > pi_4 ) { maxChange2 /= 2; }

    do {
      Axis2 = axis( best.theta, best.phi + maxChange2 * epsilon );
      // fixed odd behavoir L.L. adding epsilon
      theThrust = thrust(Axis2) + epsilon;
      if ( theThrust < c ) maxChange2 /= 2;
    } while ( theThrust < c );

    best.phi += maxChange2 * epsilon;
    if ( best.phi > pi2 ) best.phi -= pi2;
    if ( best.phi < 0 ) best.phi += pi2;

    if ( mand_ct > 0 ) mand_ct--;
    max_ct--;
    done = ( maxChange1 * maxChange1 > 1 ||
	     maxChange2 * maxChange2 > 1 ||
	     mand_ct ) && ( max_ct > 0 );
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
