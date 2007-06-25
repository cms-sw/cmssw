#ifndef PhysicsTools_IsolationUtils_ConeAreaFunction_h
#define PhysicsTools_IsolationUtils_ConeAreaFunction_h

// -*- C++ -*-
//
// Package:    ConeAreaFunction
// Class:      ConeAreaFunction
// 
/**\class ConeAreaFunction ConeAreaFunction.cc PhysicsTools/IsolationUtils/src/ConeAreaFunction.cc

 Description: low level class to compute area of signal cone
              corresponding to three-dimensional opening angle alpha given as function argument

 Implementation:
     imported into CMSSW on 05/18/2007
*/
//
// Original Author:  Christian Veelken, UC Davis
//         Created:  Thu Nov  2 13:47:40 CST 2006
// $Id: ConeAreaFunction.cc,v 1.3 2006/11/30 17:07:28 dwjang Exp $
//
//

// ROOT include files
#include <Math/ParamFunction.h>
#include <Math/Integrator.h>

class IntegrandThetaFunction;

//
// class declaration
//

class ConeAreaFunction : public ROOT::Math::ParamFunction
{
 public:
  ConeAreaFunction();
  ConeAreaFunction(const ConeAreaFunction& bluePrint);
  ~ConeAreaFunction();

  ConeAreaFunction& operator=(const ConeAreaFunction& bluePrint);

  void SetParameterTheta0(double theta0);
  void SetParameterPhi0(double phi0);

  void SetAcceptanceLimit(double etaMax);

  ROOT::Math::IGenFunction* Clone () const { return new ConeAreaFunction(*this); }

 protected:
  void SetParameters(double* param);

  double DoEval(double x) const;
  double DoDerivative(double x) const;
  void DoParameterGradient(double x, double* paramGradient) const;

// !!! ONLY FOR TESTING
double theta0_; // polar angle of cone axis
double phi0_; // azimuth angle of cone axis

double etaMax_; // maximum pseudo-rapidity at which particles used for tau cone isolation can be detected (etaMax = 2.5 for charged particles; etaMax ~ 4.0 for neutrals)
// !!! ONLY FOR TESTING

 private:
/*
  double theta0_; // polar angle of cone axis
  double phi0_; // azimuth angle of cone axis

  double etaMax_; // maximum pseudo-rapidity at which particles used for tau cone isolation can be detected (etaMax = 2.5 for charged particles; etaMax ~ 4.0 for neutrals)
*/
  mutable IntegrandThetaFunction* fTheta_;
  mutable ROOT::Math::Integrator* integrator_;

  static const unsigned int debugLevel_ = 0;
};

#endif
