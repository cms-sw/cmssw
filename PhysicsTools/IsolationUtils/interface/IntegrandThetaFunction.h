#ifndef PhysicsTools_IsolationUtils_IntegrandThetaFunction_h
#define PhysicsTools_IsolationUtils_IntegrandThetaFunction_h

// -*- C++ -*-
//
// Package:    IntegrandThetaFunction
// Class:      IntegrandThetaFunction
// 
/**\class IntegrandThetaFunction IntegrandThetaFunction.cc PhysicsTools/IsolationUtils/src/IntegrandThetaFunction.cc

 Description: auxialiary class for fixed area isolation cone computation
              (this class performs the integration over the polar angle)

 Implementation:
     imported into CMSSW on 05/18/2007
*/
//
// Original Author:  Christian Veelken, UC Davis
//         Created:  Thu Nov  2 13:47:40 CST 2006
// $Id: IntegrandThetaFunction.h,v 1.2 2009/01/14 10:53:14 hegner Exp $
//
//

// ROOT include files
#include <Math/IFunction.h>
#include <Math/IFunctionfwd.h>

// CMSSW include files
#include "PhysicsTools/IsolationUtils/interface/IntegralOverPhiFunction.h"

//
// class declaration
//

class IntegrandThetaFunction : public ROOT::Math::IGenFunction
{
 public:
  IntegrandThetaFunction();
  IntegrandThetaFunction(const IntegrandThetaFunction& bluePrint);
  virtual ~IntegrandThetaFunction();

  IntegrandThetaFunction& operator=(const IntegrandThetaFunction& bluePrint);
  
  void SetParameterTheta0(double theta0);
  void SetParameterPhi0(double phi0);
  void SetParameterAlpha(double alpha);

  virtual ROOT::Math::IGenFunction* Clone () const { return new IntegrandThetaFunction(*this); }

 private:
  double DoEval(double x) const;

  double theta0_; // polar angle of cone axis
  double phi0_; // azimuth angle of cone axis
  double alpha_; // opening angle of cone (measured from cone axis)

  mutable IntegralOverPhiFunction fPhi_;

  static const unsigned int debugLevel_ = 0;
};

#endif
