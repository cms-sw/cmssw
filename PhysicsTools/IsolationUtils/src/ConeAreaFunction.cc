#include "PhysicsTools/IsolationUtils/interface/ConeAreaFunction.h"

// -*- C++ -*-
//
// Package:    ConeAreaFunction
// Class:      ConeAreaFunction
// 
/**\class ConeAreaFunction ConeAreaFunction.cc PhysicsTools/IsolationUtils//src/ConeAreaFunction.cc

 Description: low level class to compute area of signal cone
              corresponding to three-dimensional opening angle alpha given as function argument

 Implementation:
     imported into CMSSW on 05/18/2007
*/
//
// Original Author:  Christian Veelken, UC Davis
//         Created:  Thu Nov  2 13:47:40 CST 2006
// $Id: ConeAreaFunction.cc,v 1.3 2009/01/14 10:53:15 hegner Exp $
//
//

// C++ standard library include files
#include <iostream>
#include <iomanip>
#include <vector>

// ROOT include files
#include <TMath.h>

// CMSSW include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/IsolationUtils/interface/IntegrandThetaFunction.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

//
// constructors and destructor
//

ConeAreaFunction::ConeAreaFunction()
  : ROOT::Math::ParamFunction<ROOT::Math::IParametricGradFunctionOneDim>(2)
{
  theta0_ = 0.; 
  phi0_ = 0.; 

  etaMax_ = -1;

  fTheta_ = new IntegrandThetaFunction();
  integrator_ = new ROOT::Math::Integrator(*fTheta_);
}

ConeAreaFunction::ConeAreaFunction(const ConeAreaFunction& bluePrint)
{
  theta0_ = bluePrint.theta0_;
  phi0_ = bluePrint.phi0_;

  etaMax_ = bluePrint.etaMax_;

  fTheta_ = new IntegrandThetaFunction(*bluePrint.fTheta_);
  integrator_ = new ROOT::Math::Integrator(*fTheta_);
}

ConeAreaFunction::~ConeAreaFunction()
{
  delete fTheta_; // function gets deleted automatically by Integrator ?
  delete integrator_;
}

//
// assignment operator
//

ConeAreaFunction& ConeAreaFunction::operator=(const ConeAreaFunction& bluePrint)
{
  theta0_ = bluePrint.theta0_;
  phi0_ = bluePrint.phi0_;
  
  etaMax_ = bluePrint.etaMax_;

  (*fTheta_) = (*bluePrint.fTheta_);
  integrator_->SetFunction(*fTheta_);

  return (*this);
}

//
// member functions
//

void ConeAreaFunction::SetParameterTheta0(double theta0)
{
  theta0_ = theta0;
}

void ConeAreaFunction::SetParameterPhi0(double phi0)
{
  phi0_ = normalizedPhi(phi0); // map azimuth angle into interval [-pi,+pi]
}

void ConeAreaFunction::SetParameters(double* param)
{
  if ( debugLevel_ > 0 ) {
    edm::LogVerbatim("") << "<ConeAreaFunction::SetParameters>:" << std::endl
			 << " theta0 = " << param[0] << std::endl
			 << " phi0 = " << param[1] << std::endl;
  }

  theta0_ = param[0];
  phi0_ = param[1];
}

void ConeAreaFunction::SetAcceptanceLimit(double etaMax)
{
//--- check that pseudo-rapidity given as function argument is positive
//    (assume equal acceptance for positive and negative pseudo-rapidities)

  if ( etaMax > 0 ) {
    etaMax_ = etaMax;
  } else {
    edm::LogError("") << "etaMax cannot be negative !" << std::endl;
  }
}

double ConeAreaFunction::DoEvalPar(double x, const double* param) const
{
//--- calculate area covered by cone of opening angle alpha
//    (measured from cone axis);
//    evaluate integral over angle theta
//    (polar angle of point within cone)
// FIXME: the const above is actually not true as it is implemented now.

  theta0_ = param[0];
  phi0_ = param[1];

  return DoEval(x);
}


double ConeAreaFunction::DoEval(double x) const
{
//--- calculate area covered by cone of opening angle alpha
//    (measured from cone axis);
//    evaluate integral over angle theta
//    (polar angle of point within cone)

  fTheta_->SetParameterTheta0(theta0_);
  fTheta_->SetParameterPhi0(phi0_);
  fTheta_->SetParameterAlpha(x);

  integrator_->SetFunction(*fTheta_); // set updated parameter values in Integrator

  double thetaMin = (etaMax_ > 0) ? 2*TMath::ATan(TMath::Exp(-etaMax_)) : 0.;
  double thetaMax = TMath::Pi() - thetaMin;

  double integralOverTheta = integrator_->Integral(thetaMin, thetaMax);

  return integralOverTheta;
}  

double ConeAreaFunction::DoDerivative(double x) const
{
//--- virtual function inherited from ROOT::Math::ParamFunction base class;
//    not implemented, because not neccessary, but needs to be defined to make code compile...
  edm::LogWarning("") << "Function not implemented yet !" << std::endl;

  return 0.;
}

double ConeAreaFunction::DoParameterDerivative(double, const double*, unsigned int) const
{
//--- virtual function inherited from ROOT::Math::ParamFunction base class;
//    not implemented, because not neccessary, but needs to be defined to make code compile...
  edm::LogWarning("") << "Function not implemented yet !" << std::endl;

  return 0.;
}



void ConeAreaFunction::DoParameterGradient(double x, double* paramGradient) const
{
//--- virtual function inherited from ROOT::Math::ParamFunction base class;
//    not implemented, because not neccessary, but needs to be defined to make code compile...
  edm::LogWarning("") << "Function not implemented yet !" << std::endl;
}

