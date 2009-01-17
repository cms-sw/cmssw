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

#include "DataFormats/Math/interface/normalizedPhi.h"

//
// constructors and destructor
//

ConeAreaFunction::ConeAreaFunction()
  : fTheta_()
{
  theta0_ = 0.; 
  phi0_ = 0.; 

  etaMax_ = -1;
  
  integrator_ = new ROOT::Math::Integrator(fTheta_);
}

ConeAreaFunction::ConeAreaFunction(const ConeAreaFunction& bluePrint)
{
  theta0_ = bluePrint.theta0_;
  phi0_ = bluePrint.phi0_;

  etaMax_ = bluePrint.etaMax_;

  fTheta_ = bluePrint.fTheta_;
  integrator_ = new ROOT::Math::Integrator(fTheta_);
}

ConeAreaFunction::~ConeAreaFunction()
{
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

  fTheta_ = bluePrint.fTheta_;
  integrator_->SetFunction(fTheta_);

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

double ConeAreaFunction::DoEval(double x) const
{
//--- calculate area covered by cone of opening angle alpha
//    (measured from cone axis);
//    evaluate integral over angle theta
//    (polar angle of point within cone)

  fTheta_.SetParameterTheta0(theta0_);
  fTheta_.SetParameterPhi0(phi0_);
  fTheta_.SetParameterAlpha(x);

  integrator_->SetFunction(fTheta_); // set updated parameter values in Integrator

  double thetaMin = (etaMax_ > 0) ? 2*TMath::ATan(TMath::Exp(-etaMax_)) : 0.;
  double thetaMax = TMath::Pi() - thetaMin;

  double integralOverTheta = integrator_->Integral(thetaMin, thetaMax);

  return integralOverTheta;
}  
