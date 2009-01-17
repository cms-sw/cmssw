#include "PhysicsTools/IsolationUtils/interface/IntegralOverPhiFunction.h"

// -*- C++ -*-
//
// Package:    IntegralOverPhiFunction
// Class:      IntegralOverPhiFunction
// 
/**\class IntegralOverPhiFunction IntegralOverPhiFunction.cc PhysicsTools/IsolationUtils/src/IntegralOverPhiFunction.cc

 Description: auxialiary class for fixed area isolation cone computation
              (this class performs the integration over the azimuthal angle)

 Implementation:
     imported into CMSSW on 05/18/2007
*/
//
// Original Author:  Christian Veelken, UC Davis
//         Created:  Thu Nov  2 13:47:40 CST 2006
// $Id: IntegralOverPhiFunction.cc,v 1.3 2009/01/14 10:53:15 hegner Exp $
//
//

// system include files
#include <iostream>
#include <iomanip>
#include <vector>

// ROOT include files
#include <TMath.h>

// CMSSW include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/normalizedPhi.h"

//
// constructors and destructor
//

IntegralOverPhiFunction::IntegralOverPhiFunction()
{ 
  theta0_ = 0.; 
  phi0_ = 0.; 
  alpha_ = 0.; 
}

IntegralOverPhiFunction::~IntegralOverPhiFunction()
{}

//
// member functions
//

void IntegralOverPhiFunction::SetParameterTheta0(double theta0)
{
  theta0_ = theta0;
}

void IntegralOverPhiFunction::SetParameterPhi0(double phi0)
{
  phi0_ = normalizedPhi(phi0); // map azimuth angle into interval [-pi,+pi]
}

void IntegralOverPhiFunction::SetParameterAlpha(double alpha)
{
  alpha_ = alpha;
}

double IntegralOverPhiFunction::DoEval(double x) const
{
//--- return zero if theta either close to zero or close to Pi
//    (phi not well-defined in that case;
//     numerical expressions might become "NaN"s)
  const double epsilon = 1.e-3;
  if ( x < epsilon || x > (TMath::Pi() - epsilon) ) return 0.;

//--- calculate trigonometric expressions
//    (dependend on angle theta0;
//     polar angle of cone axis);
//    avoid theta0 exactly zero or exactly Pi
//    (numerical expressions become "NaN"s)
  double theta0 = theta0_;
  if ( theta0 <                epsilon ) theta0 = epsilon;
  if ( theta0 > (TMath::Pi() - epsilon)) theta0 = (TMath::Pi() - epsilon);
  double cosTheta0 = TMath::Cos(theta0);
  double cos2Theta0 = TMath::Cos(2*theta0);
  double sinTheta0 = TMath::Sin(theta0);
  double cotTheta0 = 1./TMath::Tan(theta0);
  double cscTheta0 = 1./TMath::Sin(theta0);
//    (dependend on angle phi0;
//     azimuth angle of cone axis)
//    avoid phi0 exactly -Pi/2 or exactly +Pi/2
//    (numerical expressions become ambiguous)
  double phi0 = phi0_;
  if ( phi0 >  (-TMath::Pi()/2 - epsilon) && phi0 <  -TMath::Pi()/2            ) phi0 = -TMath::Pi()/2 - epsilon;
  if ( phi0 >=  -TMath::Pi()/2            && phi0 < (-TMath::Pi()/2 + epsilon) ) phi0 = -TMath::Pi()/2 + epsilon;
  if ( phi0 >  ( TMath::Pi()/2 - epsilon) && phi0 <   TMath::Pi()/2            ) phi0 =  TMath::Pi()/2 - epsilon;
  if ( phi0 >=   TMath::Pi()/2            && phi0 < ( TMath::Pi()/2 + epsilon) ) phi0 =  TMath::Pi()/2 + epsilon;
  double cosPhi0 = TMath::Cos(phi0);
  double tanPhi0 = TMath::Tan(phi0);
//    (dependend on angle theta;
//     polar angle of point within cone)
  double cosTheta = TMath::Cos(x);
  double cos2Theta = TMath::Cos(2*x);
  double sinTheta = TMath::Sin(x);
  double cotTheta = 1./TMath::Tan(x);
  double cscTheta = 1./TMath::Sin(x);
//    (dependent on angle alpha;
//     opening angle of cone, measured from cone axis)
  double cosAlpha = TMath::Cos(alpha_);

  double s = -cosPhi0*cosPhi0*(2*cosAlpha*cosAlpha + cos2Theta - 4*cosAlpha*cosTheta*cosTheta0 + cos2Theta0)*sinTheta*sinTheta*sinTheta0*sinTheta0;

//--- return either zero or 2 Pi
//    in case argument of square-root is zero or negative
//    (negative values solely arise from rounding errors);
//    check whether to return zero or 2 Pi:
//     o return zero 
//       if |theta- theta0| > alpha, 
//       (theta outside cone of opening angle alpha, so vanishing integral over phi)
//     o return 2 Pi
//       if |theta- theta0| < alpha
//       (theta within cone of opening angle alpha;
//        actually theta0 < alpha in forward/backward direction,
//        so that integral includes all phi values, hence yielding 2 Pi)
  if ( s <= 0 ) {
    if ( TMath::Abs(x - theta0) >  alpha_ ) return 0;
    if ( TMath::Abs(x - theta0) <= alpha_ ) return 2*TMath::Pi();
    edm::LogError("") << "Failed to compute return value !";
  }

  double r = (1./TMath::Sqrt(2.))*(cscTheta*cscTheta*cscTheta0*cscTheta0*TMath::Sqrt(s)*tanPhi0);
  double t = cosPhi0*(-cotTheta*cotTheta0 + cosAlpha*cscTheta*cscTheta0);
  
  double phi[4];
  phi[0] = -TMath::ACos(t - r);
  phi[1] =  TMath::ACos(t - r);
  phi[2] = -TMath::ACos(t + r);
  phi[3] =  TMath::ACos(t + r);

  if ( debugLevel_ > 0 ) {
    edm::LogVerbatim("") << "phi0 = " << phi0 << std::endl
			 << "phi[0] = " << phi[0] << " (phi[0] - phi0 = " << (phi[0] - phi0)*180/TMath::Pi() << ")" << std::endl
			 << "phi[1] = " << phi[1] << " (phi[1] - phi0 = " << (phi[1] - phi0)*180/TMath::Pi() << ")" << std::endl
			 << "phi[2] = " << phi[2] << " (phi[2] - phi0 = " << (phi[2] - phi0)*180/TMath::Pi() << ")" << std::endl
			 << "phi[3] = " << phi[3] << " (phi[3] - phi0 = " << (phi[3] - phi0)*180/TMath::Pi() << ")" << std::endl;
  }

  double phiMin = 0.;
  double phiMax = 0.;  
  for ( unsigned int i = 0; i < 4; ++i ) {
    for ( unsigned int j = (i + 1); j < 4; ++j ) {
//--- search for the two solutions for phi
//    that have an equal distance to phi0
//    and are on either side
      double dPhi_i = phi[i] - phi0;
      double dPhi_j = phi0 - phi[j];
      if ( TMath::Abs(normalizedPhi(dPhi_i - dPhi_j)) < epsilon ) { // map difference in azimuth angle into interval [-pi,+pi] and require it to be negligible
	phiMin = TMath::Min(phi[i], phi[j]);
	phiMax = TMath::Max(phi[i], phi[j]);
      }
    }
  }

  if ( phiMin == 0 && phiMax == 0 ) {
    edm::LogError("") << "failed to compute Return Value !" << std::endl;
  }

  return TMath::Abs(normalizedPhi(phi0 - phiMin)) + TMath::Abs(normalizedPhi(phiMax - phi0));
}  
