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
//
//

// ROOT include files
#include <Math/ParamFunction.h>
#include <Math/Integrator.h>

class IntegrandThetaFunction;

//
// class declaration
//

class ConeAreaFunction : public ROOT::Math::ParamFunction<ROOT::Math::IParametricGradFunctionOneDim>
{
 public:
  ConeAreaFunction();
  ConeAreaFunction(const ConeAreaFunction& bluePrint);
  ~ConeAreaFunction();

  ConeAreaFunction& operator=(const ConeAreaFunction& bluePrint);

  void SetParameterTheta0(double theta0);
  void SetParameterPhi0(double phi0);

  void SetAcceptanceLimit(double etaMax);

  virtual ROOT::Math::IGenFunction* Clone () const override { return new ConeAreaFunction(*this); }

 protected:
  void SetParameters(double const* param) override;
  virtual double DoEvalPar(double , const double *) const override;
  double DoEval(double x) const override;
  double DoDerivative(double x) const;
  void DoParameterGradient(double x, double* paramGradient) const;
  virtual double DoParameterDerivative(double, const double*, unsigned int) const override;

// !!! ONLY FOR TESTING
mutable double theta0_; // polar angle of cone axis
mutable double phi0_; // azimuth angle of cone axis

mutable double etaMax_; // maximum pseudo-rapidity at which particles used for tau cone isolation can be detected (etaMax = 2.5 for charged particles; etaMax ~ 4.0 for neutrals)
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
