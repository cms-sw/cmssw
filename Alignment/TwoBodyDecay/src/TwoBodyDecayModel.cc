
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"


TwoBodyDecayModel::TwoBodyDecayModel( double mPrimary, double mSecondary ) :
  thePrimaryMass( mPrimary ), theSecondaryMass( mSecondary ) {}


TwoBodyDecayModel::~TwoBodyDecayModel() {}


AlgebraicMatrix TwoBodyDecayModel::rotationMatrix( double px, double py, double pz )
{
  // compute transverse and absolute momentum
  double pT2 = px*px + py*py;
  double p2 = pT2 + pz*pz;
  double pT = sqrt( pT2 );
  double p = sqrt( p2 );

  AlgebraicMatrix rotMat( 3, 3 );

  // compute rotation matrix
  rotMat[0][0] = px*pz/pT/p;
  rotMat[0][1] = -py/pT;
  rotMat[0][2] = px/p;

  rotMat[1][0] = py*pz/pT/p;
  rotMat[1][1] = px/pT;
  rotMat[1][2] = py/p;

  rotMat[2][0] = -pT/p;
  rotMat[2][1] = 0.;
  rotMat[2][2] = pz/p;

  return rotMat;
}


AlgebraicMatrix TwoBodyDecayModel::curvilinearToCartesianJacobian( double rho, double theta, double phi, double zMagField )
{
  double q = ( ( rho < 0 ) ? -1. : 1. );
  double conv = q*zMagField;

  double stheta = sin( theta );
  double ctheta = cos( theta );
  double sphi = sin( phi );
  double cphi = cos( phi );

  AlgebraicMatrix curv2cart( 3, 3 );

  curv2cart[0][0] = -rho*cphi;
  curv2cart[0][1] = -rho*sphi;
  curv2cart[0][2] = 0.;

  curv2cart[1][0] = cphi*stheta*ctheta;
  curv2cart[1][1] = sphi*stheta*ctheta;
  curv2cart[1][2] = -stheta*stheta;

  curv2cart[2][0] = -sphi;
  curv2cart[2][1] = cphi;
  curv2cart[2][2] = 0.;

  curv2cart *= rho/conv;

  return curv2cart;
}


AlgebraicMatrix TwoBodyDecayModel::curvilinearToCartesianJacobian( AlgebraicVector curv, double zMagField )
{
  return this->curvilinearToCartesianJacobian( curv[0], curv[1], curv[2], zMagField );
}


AlgebraicVector TwoBodyDecayModel::convertCurvilinearToCartesian( AlgebraicVector curv, double zMagField )
{
  double rt = fabs( zMagField/curv[0] );

  AlgebraicVector cart( 3 );
  cart[0] = rt*cos( curv[2] );
  cart[1] = rt*sin( curv[2] );
  cart[2] = rt/tan( curv[1] );

  return cart;
}


const std::pair< AlgebraicVector, AlgebraicVector > TwoBodyDecayModel::cartesianSecondaryMomenta( const AlgebraicVector & param )
{
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  // compute transverse and absolute momentum
  double pT2 = px*px + py*py;
  double p2 = pT2 + pz*pz;
  double p = sqrt( p2 );

  double sphi = sin( phi );
  double cphi = cos( phi );
  double stheta = sin( theta );
  double ctheta = cos( theta );

  // some constants from kinematics
  double c1 = 0.5*thePrimaryMass/theSecondaryMass;
  double c2 = sqrt( c1*c1 - 1. );
  double c3 = 0.5*c2*ctheta/c1;
  double c4 = sqrt( p2 + thePrimaryMass*thePrimaryMass );

  // momentum of decay particle 1 in the primary's boosted frame
  AlgebraicMatrix pplus( 3, 1 );
  pplus[0][0] = theSecondaryMass*c2*stheta*cphi;
  pplus[1][0] = theSecondaryMass*c2*stheta*sphi;
  pplus[2][0] = 0.5*p + c3*c4;

  // momentum of decay particle 2 in the primary's boosted frame
  AlgebraicMatrix pminus( 3, 1 );
  pminus[0][0] = -pplus[0][0];
  pminus[1][0] = -pplus[1][0];
  pminus[2][0] = 0.5*p - c3*c4;

  AlgebraicMatrix rotMat = rotationMatrix( px, py, pz );

  return std::make_pair( rotMat*pplus, rotMat*pminus );
}


const std::pair< AlgebraicVector, AlgebraicVector > TwoBodyDecayModel::cartesianSecondaryMomenta( const TwoBodyDecay & tbd )
{
  return cartesianSecondaryMomenta( tbd.parameters() );
}

const std::pair< AlgebraicVector, AlgebraicVector > TwoBodyDecayModel::cartesianSecondaryMomenta( const TwoBodyDecayParameters & tbdparam )
{
  return cartesianSecondaryMomenta( tbdparam.parameters() );
}
