#include <cmath>
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "DataFormats/Math/interface/LorentzVector.h"
using namespace std;
using namespace aod;
typedef math::XYZTLorentzVector LorentzVector;

// constants global to this file
const double epsilon = 0.0001;
const int nSegsTheta = 10; // number of initial segments in theta
const int nSegsPhi = 10; // number of initial segments in phi
const int nSegs = nSegsTheta * nSegsPhi; // total number of segments

Thrust::Thrust( const_iterator begin, const_iterator end ) 
  : _thrust(0), _axis( 0, 0, 0 ), _denom_sum(0), n_( end - begin ) {
  if (n_>0) { // make sure there is at least one track
    // copy momentum components to local variables for speed
    vector<double> valX(n_), valY(n_), valZ(n_);
    int i = 0;
    for( const_iterator trkptr = begin; trkptr != end; ++trkptr ) {
      LorentzVector cmList(trkptr->p4());
      valX[i] = Vector(cmList).x();
      valY[i] = Vector(cmList).y();
      valZ[i] = Vector(cmList).z();
      _denom_sum += Vector(cmList).r();
      i++;
    } // end while

    // get initial estimate for thrust axis
    vector<double> init_axis = get_initial_axis(valX,valY,valZ);
    // use initial estimate to get more refined axis
    vector<double> final_axis = get_final_axis(init_axis[0],init_axis[1],valX,valY,
                            valZ);
    // convert theta,phi to x,y,z
    vector<double> theAxis = get_axis(final_axis[0],final_axis[1]);
    // set axis
    _axis = Vector( theAxis[0], theAxis[1], theAxis[2] ); 
    if ( cos( _axis.theta() ) < 0 ) 
      {
	_axis *= -1;
      } // reverse
    // set thrust
    _thrust = calc_thrust( theAxis, valX, valY, valZ );
  } // end if n_>0
}

// return theta, phi initial
vector<double> Thrust::get_initial_axis(const vector<double> & valX, 
					const vector<double> & valY,
					const vector<double> & valZ ) const {

  int i,j;
  // get initial estimate for thrust axis
  vector<double> rInitial(3); // 3 dimensions
  double thrust[nSegs], max=0;
  int indI=0,indJ=0,index = -1; // index to max

  // calculate the thrust in different theta, phi regions
  for (i=0;i<nSegsTheta;i++) { // loop over theta
    double z = cos( M_PI * i / (nSegsTheta-1) ); // z
    double invZ = sqrt(1-z*z);
    for (j=0;j<nSegsPhi;j++) { // loop over phi
      rInitial[0] = invZ * cos( 2 * M_PI * j / nSegsPhi ); // x
      rInitial[1] = invZ * sin( 2 * M_PI * j / nSegsPhi ); // x
      rInitial[2] = z;
      // get thrust for this axis
      thrust[i*nSegsPhi+j] = calc_thrust(rInitial,valX,valY,valZ);
      if (thrust[i*nSegsPhi+j]>max) {
        index = i*nSegsPhi+j;
        indI = i;
	indJ = j;
        max = thrust[index];
      } // end if thrust>max
    } // end for i
  } // end for j

  // take max and one point on either size, fitting to a parabola and
  // extrapolating to the real max.  Do this separately for each dimension.
  // y = ax^2 + bx + c.  At the max, x = 0, on either side, x = +/-1.
  // do phi first
  double a, b, c=max;
  int ind1 = indJ + 1;
  if (ind1>=nSegsPhi)
    ind1 -= nSegsPhi;
  int ind2 = indJ - 1;
  if (ind2<0)
    ind2 += nSegsPhi;
  a = ( thrust[ind1]+thrust[ind2]-2*c ) / 2;
  b = thrust[ind1] - a - c;
  double maxPhiInd = 0; // avoid divide by 0
  if (a!=0)
    maxPhiInd = -b/(2*a); // max 
  // do theta
  double maxThetaInd;
  if (indI==0||indI==(nSegsTheta-1)) // special case of end points
    maxThetaInd = indI;
  else { // not at end point
    ind1 = indI + 1;
    ind2 = indI - 1;
    a = ( thrust[ind1] + thrust[ind2] - 2*c ) / 2;
    b = thrust[ind1] - a - c; 
    maxThetaInd = 0; // avoid divide by 0
    if (a!=0)
      maxThetaInd = -b/(2*a); // max
  } // end else

  // calculate best guess at theta and phi then get starting point to iterate
  double bestTheta = M_PI * (maxThetaInd+indI) / (nSegsTheta-1);
  double bestPhi = 2 * M_PI * (maxPhiInd+indJ) / nSegsPhi;

  // set return values
  vector<double> result( 2 );
  result[0] = bestTheta;
  result[1] = bestPhi;
  return result;
} // end get_initial_axis

// get the final axis to be used
vector<double> Thrust::get_final_axis(double bestTheta, double bestPhi,
				      const vector<double> & valX, 
				      const vector<double> & valY,
				      const vector<double> & valZ ) const {

  double maxChange1,maxChange2;
  double a,b,c;
  double theThrust;
  int mand_ct = 3; // mandatory number of passes
  int max_ct = 1000; // a very large number of iterations

  // loop until done
  int done;
  do { 
    // get three axis to estimate maximum
    vector<double> theAxis = get_axis(bestTheta,bestPhi);
    vector<double> Axis2 = get_axis(bestTheta+epsilon,bestPhi); // do differential
    vector<double> Axis3 = get_axis(bestTheta-epsilon,bestPhi); // do differential

    // use parabolic approx as above
    c = calc_thrust(theAxis,valX,valY,valZ);
    a = ( calc_thrust(Axis2,valX,valY,valZ) - 2 * c +
          calc_thrust(Axis3,valX,valY,valZ) ) / 2;
    b = calc_thrust(Axis2,valX,valY,valZ) - a - c; 

    // calculate max
    maxChange1 = 10 * ( b<0 ? -1 : 1 ); // linear 
    if (a!=0)
      maxChange1 = -b/(2*a);

    // make sure change is small to avoid convergence problems
    while (fabs(maxChange1*epsilon)>M_PI/4) {maxChange1 /= 2;} // small changes

    // special case, use a different phi
    if (maxChange1==0&&(bestTheta==0||bestTheta==M_PI)) { 
      bestPhi += M_PI_2;
      if (bestPhi>2*M_PI)
        bestPhi -= 2 * M_PI;

      // get three axis to estimate maximum
      theAxis = get_axis(bestTheta,bestPhi);
      Axis2 = get_axis(bestTheta+epsilon,bestPhi); // do differential
      Axis3 = get_axis(bestTheta-epsilon,bestPhi); // do differential

      // use parabolic approx as above
      c = calc_thrust(theAxis,valX,valY,valZ);
      a = ( calc_thrust(Axis2,valX,valY,valZ) - 2 * c +
            calc_thrust(Axis3,valX,valY,valZ) ) / 2;
      b = calc_thrust(Axis2,valX,valY,valZ) - a - c;

      // calculate max
      maxChange1 = 10 * ( b<0 ? -1 : 1 ); // linear 
      if (a!=0)
        maxChange1 = -b/(2*a);

    } // end special case

    // loop until change is at least good enough as started with
    do {
      Axis2 = get_axis(bestTheta+maxChange1*epsilon,bestPhi);
      theThrust = calc_thrust(Axis2,valX,valY,valZ);
      if (theThrust<c)
        maxChange1 /= 2; // don't trust large jumps so be willing to reduce

    } while (theThrust<c); // do until at least equal 

    // put bestTheta in correct units and check for consistency
    bestTheta += maxChange1 * epsilon;
    if (bestTheta>M_PI) {
      bestTheta = M_PI - (bestTheta-M_PI);
      bestPhi += M_PI;
      if (bestPhi>2*M_PI)
        bestPhi -= 2 * M_PI;
    } // end if >pi
    if (bestTheta<0) {
      bestTheta *= -1; // change sign
      bestPhi += M_PI;
      if (bestPhi>2*M_PI)
        bestPhi -= 2 * M_PI;
    } // end if <0

    // do again for phi
    theAxis = get_axis(bestTheta,bestPhi);
    Axis2 = get_axis(bestTheta,bestPhi+epsilon); // do differential
    Axis3 = get_axis(bestTheta,bestPhi-epsilon); // do differential

    // use parabolic approx as above
    c = calc_thrust(theAxis,valX,valY,valZ);
    a = ( calc_thrust(Axis2,valX,valY,valZ) - 2 * c +
          calc_thrust(Axis3,valX,valY,valZ) ) / 2;
    b = calc_thrust(Axis2,valX,valY,valZ) - a - c;

    // get maximum
    maxChange2 = 10 * ( b<0 ? -1 : 1 ); // linear 
    if (a!=0)
      maxChange2 = -b/(2*a);

    // require small change
    while (fabs(maxChange2*epsilon)>M_PI/4) { maxChange2 /= 2; }

    // loop until change is at least as good as started with
    do {
      Axis2 = get_axis(bestTheta,bestPhi+maxChange2*epsilon);
      theThrust = calc_thrust(Axis2,valX,valY,valZ);
      if (theThrust<c)
        maxChange2 /= 2; // don't trust large jumps so be willing to reduce

    } while (theThrust<c); // do until at least equal

    // put bestPhi in correct units and check for consistency
    bestPhi += maxChange2 * epsilon;
    if (bestPhi>2*M_PI)
      bestPhi -= 2 * M_PI;
    if (bestPhi<0)
      bestPhi += 2 * M_PI;

    // update mandatory count
    if (mand_ct>0)
      mand_ct--;
    max_ct--;
    done = (maxChange1*maxChange1>1||maxChange2*maxChange2>1||mand_ct)
           &&(max_ct>0);
  } while (done);
  // continue until both are less than one epsilon

  // set return values
  vector<double> result( 2 );
  result[0] = bestTheta;
  result[1] = bestPhi;
  return result;
} // end of get_final_axis

// get x,y,z from theta phi
vector<double> Thrust::get_axis(double theta, double phi) const {

  vector<double> result( 3 );
  result[2] = cos(theta); // z
  double theSin = sin(theta); // sin calculation is slow
  result[0] = theSin * cos(phi); // x
  result[1] = theSin * sin(phi); // y

  return result;
}  // end of get_axis

// calculate the thrust for a given axis and set of three vectors
double Thrust::calc_thrust(const vector<double> & axis, 
			   const vector<double> & valX, 
			   const vector<double> & valY,
			   const vector<double> & valZ) const {
  
  double result = 0;
  double num_sum = 0;

  for (unsigned int i=0;i<n_;i++)
    // calculate sum of dot products
    num_sum += fabs( axis[0]*valX[i] + axis[1]*valY[i] + axis[2]*valZ[i] );

  // set return value
  if (_denom_sum>0) 
    result = num_sum/_denom_sum;
  return result;
}
