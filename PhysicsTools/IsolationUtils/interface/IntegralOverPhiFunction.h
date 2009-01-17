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
// $Id: IntegralOverPhiFunction.h,v 1.2 2009/01/14 10:53:14 hegner Exp $
//
//

// ROOT include files
#include <Math/IFunction.h>
#include <Math/IFunctionfwd.h>

//
// class declaration
//

class IntegralOverPhiFunction : public ROOT::Math::IGenFunction
{
 public:
  IntegralOverPhiFunction();
  virtual ~IntegralOverPhiFunction();

  void SetParameterTheta0(double theta0);
  void SetParameterPhi0(double phi0);
  void SetParameterAlpha(double alpha);

  virtual ROOT::Math::IGenFunction* Clone () const { return new IntegralOverPhiFunction(*this); }

 private:
  double DoEval(double x) const;

  double theta0_; // polar angle of cone axis
  double phi0_; // azimuth angle of cone axis
  double alpha_; // opening angle of cone (measured from cone axis)

  static const unsigned int debugLevel_ = 0;
};

#endif
