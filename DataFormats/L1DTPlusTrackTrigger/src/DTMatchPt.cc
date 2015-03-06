/*! \class DTBtiTrigger
 *  \author Ignazio Lazzizzera
 *  \brief container to store momentum information
 *  \date 2010, Apr 2
 */

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatchPt.h"

/// Method to get the curvature radius
void DTMatchPt::findCurvatureRadius( float aMinRInvB, float aMaxRInvB,
                                     std::vector< GlobalPoint > aPosVec )
{
  if ( aPosVec.size() != 3 )
  {
    theRB = NAN;
    theRInvB = NAN;
    return;
  }

  /// Here the vector has the right size
  double X0 = aPosVec.at(0).x();
  double X1 = aPosVec.at(1).x();
  double X2 = aPosVec.at(2).x();
  double Y0 = aPosVec.at(0).y();
  double Y1 = aPosVec.at(1).y();
  double Y2 = aPosVec.at(2).y();

  double L1 = sqrt( (X0-X1)*(X0-X1) + (Y0-Y1)*(Y0-Y1) );
  double L2 = sqrt( (X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2) );
  double L1L2 = (X0-X1)*(X1-X2) + (Y0-Y1)*(Y1-Y2);
  double bendingCos = L1L2 / (L1*L2);
  double bending = acos( bendingCos );

  theRInvB = 2. * static_cast< float >( bending / (L1+L2) );

  if( theRInvB < aMinRInvB ||
      theRInvB > aMaxRInvB )
  {
    theRB = NAN;
    theRInvB = NAN;
    return;
  }

  theRB = static_cast< float >( 0.5 * (L1+L2) / bending );
}

/// Method to get the Pt
void DTMatchPt::findPt( float aMinRInvB, float aMaxRInvB,
                        std::vector< GlobalPoint > aPosVec,
                        float const aCorr )
{
  this->findCurvatureRadius( aMinRInvB, aMaxRInvB, aPosVec );

  if ( isnan( theRB ) )
  {
    return;
  }

  thePt = 0.003 * theRB * 3.8;
  thePt += aCorr * thePt;
  thePtInv = 1. / thePt;

  return;
}

/// Method to get the Pt and not only the Pt
void DTMatchPt::findPtAndParameters( float aMinRInvB, float aMaxRInvB,
                                     std::vector< GlobalPoint > aPosVec,
                                     float const aCorr )
{
  double r0   = aPosVec.at(0).perp();
  double phi0 = aPosVec.at(0).phi();
  double r1   = aPosVec.at(1).perp();
  double phi1 = aPosVec.at(1).phi();
  double r2   = aPosVec.at(2).perp();
  double phi2 = aPosVec.at(2).phi();

  if ( phi0 < 0 )
  {
    phi0 += 2. * M_PI;
  }
  if ( phi0 >= 2. * M_PI )
  {
    phi0 -= 2. * M_PI;
  }

  if ( phi1 < 0 )
  {
    phi1 += 2. * M_PI;
  }
  if ( phi1 >= 2. * M_PI )
  {
    phi1 -= 2. * M_PI;
  }

  if ( phi2 < 0 )
  {
    phi2 += 2. * M_PI;
  }
  if ( phi2 >= 2. * M_PI )
  {
    phi2 -= 2. * M_PI;
  }

  double Delta = r0*r0*(r1-r2) + r1*r1*(r2-r0) + r2*r2*(r0-r1);
  if ( Delta == 0. || Delta == NAN )
  {
    return;
  }
  double invDelta = 1/Delta;

  theRInvB = -2 * ( phi0*r0*(r1-r2) + phi1*r1*(r2-r0) + phi2*r2*(r0-r1) ) * invDelta;

  short charge = (theRInvB > 0.) - (theRInvB < 0.);

  theRInvB = fabs(theRInvB);

  if( theRInvB < aMinRInvB ||
      theRInvB > aMaxRInvB )
  {
    theRB = NAN;
    theRInvB = NAN;
    theAlpha0 = NAN;
    theD = NAN;
    return;
  }

  theRB = static_cast< float >( 1. / theRInvB );
  thePt = 0.003 * theRB * 3.8;
  thePt += aCorr * thePt;
  thePtInv = 1./thePt;

  theAlpha0 = charge * invDelta *
              ( phi0*r0*(r1*r1-r2*r2) + phi1*r1*(r2*r2-r0*r0) + phi2*r2*(r0*r0-r1*r1) );

  theD = charge * r0 * r1 * r2 * ( phi0*(r1-r2) + phi1*(r2-r0) + phi2*(r0-r1) ) * invDelta;

  return;
}

