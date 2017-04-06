#ifndef PhysicsTools_IsolationUtils_ConeAreaRootFunction_h
#define PhysicsTools_IsolationUtils_ConeAreaRootFunction_h

// -*- C++ -*-
//
// Package:    ConeAreaRootFunction
// Class:      ConeAreaRootFunction
// 
/**\class ConeAreaRootFunction ConeAreaRootFunction.cc PhysicsTools/IsolationUtils/src/ConeAreaRootFunction.cc

 Description: low level class to compute three-dimensional opening angle of isolation cone
              corresponding to area given as function argument

 Implementation:
     imported into CMSSW on 05/18/2007
*/
//
// Original Author:  Christian Veelken, UC Davis
//         Created:  Thu Nov  2 13:47:40 CST 2006
//
//

#include "PhysicsTools/IsolationUtils/interface/ConeAreaFunction.h"

//
// class declaration
//

class ConeAreaRootFunction : public ConeAreaFunction
{
 public:
  ConeAreaRootFunction();
  ConeAreaRootFunction(const ConeAreaRootFunction& bluePrint);
  ~ConeAreaRootFunction();
  
  ConeAreaRootFunction& operator=(const ConeAreaRootFunction& bluePrint);

  void SetParameterConeArea(double coneArea);

  ROOT::Math::IGenFunction* Clone () const override { return new ConeAreaRootFunction(*this); }

 private:
  void SetParameters(const double* param) override;

  double DoEval(double x) const override;

  double coneArea_; // area covered by cone

  static const unsigned int debugLevel_ = 0;
};

#endif
