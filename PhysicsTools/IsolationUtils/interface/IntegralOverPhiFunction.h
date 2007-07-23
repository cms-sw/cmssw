#ifndef PhysicsTools_IsolationUtils_IntegralOverPhiFunction_h
#define PhysicsTools_IsolationUtils_IntegralOverPhiFunction_h

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
// $Id: IntegralOverPhiFunction.cc,v 1.3 2006/11/30 17:07:28 dwjang Exp $
//
//

// ROOT include files
#include <Math/ParamFunction.h>

//
// class declaration
//

class IntegralOverPhiFunction : public ROOT::Math::ParamFunction
{
 public:
  IntegralOverPhiFunction();
  ~IntegralOverPhiFunction();

  void SetParameterTheta0(double theta0);
  void SetParameterPhi0(double phi0);
  void SetParameterAlpha(double alpha);

  ROOT::Math::IGenFunction* Clone () const { return new IntegralOverPhiFunction(*this); }

 private:
  void SetParameters(double* param);

  double DoEval(double x) const;
  double DoDerivative(double x) const;
  void DoParameterGradient(double x, double* paramGradient) const;

  double theta0_; // polar angle of cone axis
  double phi0_; // azimuth angle of cone axis
  double alpha_; // opening angle of cone (measured from cone axis)

// !!! ONLY FOR TESTING
  mutable unsigned int numSolutionMin1_;
  mutable unsigned int numSolutionMax1_;
  mutable unsigned int numSolutionMin2_;
  mutable unsigned int numSolutionMax2_;
  mutable unsigned int numSolutionMin3_;
  mutable unsigned int numSolutionMax3_;
  mutable unsigned int numSolutionMin4_;
  mutable unsigned int numSolutionMax4_;
//     FOR TESTING ONLY !!!

  static const unsigned int debugLevel_ = 0;
};

#endif
