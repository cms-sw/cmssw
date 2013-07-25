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
// declaration of auxiliary functions
//

void checkSolutions(unsigned int i, unsigned int& numSolution1, unsigned int& numSolution2, unsigned int& numSolution3, unsigned int& numSolution4);

//
// constructors and destructor
//

IntegralOverPhiFunction::IntegralOverPhiFunction()
  : ROOT::Math::ParamFunction<ROOT::Math::IParametricGradFunctionOneDim>(3)
{ 
  theta0_ = 0.; 
  phi0_ = 0.; 
  alpha_ = 0.; 

// !!! ONLY FOR TESTING
  numSolutionMin1_ = 0;
  numSolutionMax1_ = 0;
  numSolutionMin2_ = 0;
  numSolutionMax2_ = 0;
  numSolutionMin3_ = 0;
  numSolutionMax3_ = 0;
  numSolutionMin4_ = 0;
  numSolutionMax4_ = 0;
//     FOR TESTING ONLY !!!
}

IntegralOverPhiFunction::~IntegralOverPhiFunction()
{
// !!! ONLY FOR TESTING
  if ( debugLevel_ > 0 ) {
    edm::LogVerbatim("") << "<IntegralOverPhiFunction::~IntegralOverPhiFunction>:" << std::endl
			 << " numSolutionMin1 = " << numSolutionMin1_ << std::endl
			 << " numSolutionMax1 = " << numSolutionMax1_ << std::endl
			 << " numSolutionMin2 = " << numSolutionMin2_ << std::endl
			 << " numSolutionMax2 = " << numSolutionMax2_ << std::endl
			 << " numSolutionMin3 = " << numSolutionMin3_ << std::endl
			 << " numSolutionMax3 = " << numSolutionMax3_ << std::endl
			 << " numSolutionMin4 = " << numSolutionMin4_ << std::endl
			 << " numSolutionMax4 = " << numSolutionMax4_ << std::endl;
  }
//     FOR TESTING ONLY !!!
}

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

void IntegralOverPhiFunction::SetParameters(double* param)
{
  theta0_ = param[0];
  phi0_ = param[1];
  alpha_ = param[2];
}

double IntegralOverPhiFunction::DoEvalPar(double x, const double* param) const  //FIXME: in the current implementation const is not entirely true
{
  theta0_ = param[0];
  phi0_ = param[1];
  alpha_ = param[2];

  return DoEval(x);
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
  double cosPhi0 = TMath::Cos(phi0_);
  double tanPhi0 = TMath::Tan(phi0_);
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
    std::cerr << "Error in <IntegralOverPhiFunction::operator()>: failed to compute return value !" << std::endl;
  }

  double r = (1./TMath::Sqrt(2.))*(cscTheta*cscTheta*cscTheta0*cscTheta0*TMath::Sqrt(s)*tanPhi0);
  double t = cosPhi0*(-cotTheta*cotTheta0 + cosAlpha*cscTheta*cscTheta0);
  
  double phi[4];
  phi[0] = -TMath::ACos(t - r);
  phi[1] =  TMath::ACos(t - r);
  phi[2] = -TMath::ACos(t + r);
  phi[3] =  TMath::ACos(t + r);

  if ( debugLevel_ > 0 ) {
    edm::LogVerbatim("") << "phi0 = " << phi0_ << std::endl
			 << "phi[0] = " << phi[0] << " (phi[0] - phi0 = " << (phi[0] - phi0_)*180/TMath::Pi() << ")" << std::endl
			 << "phi[1] = " << phi[1] << " (phi[1] - phi0 = " << (phi[1] - phi0_)*180/TMath::Pi() << ")" << std::endl
			 << "phi[2] = " << phi[2] << " (phi[2] - phi0 = " << (phi[2] - phi0_)*180/TMath::Pi() << ")" << std::endl
			 << "phi[3] = " << phi[3] << " (phi[3] - phi0 = " << (phi[3] - phi0_)*180/TMath::Pi() << ")" << std::endl;
  }

  double phiMin = 0.;
  double phiMax = 0.;  
  for ( unsigned int i = 0; i < 4; ++i ) {
    for ( unsigned int j = (i + 1); j < 4; ++j ) {
//--- search for the two solutions for phi
//    that have an equal distance to phi0
//    and are on either side
      double dPhi_i = phi[i] - phi0_;
      double dPhi_j = phi0_ - phi[j];
      if ( TMath::Abs(normalizedPhi(dPhi_i - dPhi_j)) < epsilon ) { // map difference in azimuth angle into interval [-pi,+pi] and require it to be negligible
      //if ( TMath::Abs((phi[i] - phi0_) - (phi0_ - phi[j])) < epsilon ) { // map difference in azimuth angle into interval [-pi,+pi] and require it to be negligible
	phiMin = TMath::Min(phi[i], phi[j]);
	phiMax = TMath::Max(phi[i], phi[j]);

// !!! ONLY FOR TESTING
	if ( phi[i] == phiMin ) checkSolutions(i, numSolutionMin1_, numSolutionMin2_, numSolutionMin3_, numSolutionMin4_);
	if ( phi[i] == phiMax ) checkSolutions(i, numSolutionMax1_, numSolutionMax2_, numSolutionMax3_, numSolutionMax4_);
	if ( phi[j] == phiMin ) checkSolutions(j, numSolutionMin1_, numSolutionMin2_, numSolutionMin3_, numSolutionMin4_);
	if ( phi[j] == phiMax ) checkSolutions(j, numSolutionMax1_, numSolutionMax2_, numSolutionMax3_, numSolutionMax4_);
//     FOR TESTING ONLY !!!
      }
    }
  }

  if ( phiMin == 0 && phiMax == 0 ) {
    edm::LogError("") << "failed to compute Return Value !" << std::endl;
  }

  return TMath::Abs(normalizedPhi(phi0_ - phiMin)) + TMath::Abs(normalizedPhi(phiMax - phi0_));
}  

double IntegralOverPhiFunction::DoDerivative(double x) const
{
//--- virtual function inherited from ROOT::Math::ParamFunction base class;
//    not implemented, because not neccessary, but needs to be defined to make code compile...
  edm::LogWarning("") << "Function not implemented yet !" << std::endl;

  return 0.;
}

double IntegralOverPhiFunction::DoParameterDerivative(double, const double*, unsigned int) const
{
//--- virtual function inherited from ROOT::Math::ParamFunction base class;
//    not implemented, because not neccessary, but needs to be defined to make code compile...
  edm::LogWarning("") << "Function not implemented yet !" << std::endl;

  return 0.;
}

void IntegralOverPhiFunction::DoParameterGradient(double x, double* paramGradient) const
{
//--- virtual function inherited from ROOT::Math::ParamFunction base class;
//    not implemented, because not neccessary, but needs to be defined to make code compile...
  edm::LogWarning("") << "Function not implemented yet !" << std::endl;
}

//
// definition of auxiliary functions
//

void checkSolutions(unsigned int i, unsigned int& numSolution1, unsigned int& numSolution2, unsigned int& numSolution3, unsigned int& numSolution4)
{
  switch ( i ) {
  case 0 : 
    ++numSolution1;
    break;
  case 1 : 
    ++numSolution2;
    break;
  case 2 : 
    ++numSolution3;
    break;
  case 3 : 
    ++numSolution4;
    break;
  }
}
