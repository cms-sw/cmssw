#include "PhysicsTools/IsolationUtils/interface/ConeAreaRootFunction.h"

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
// $Id: ConeAreaRootFunction.cc,v 1.1 2007/05/23 20:21:37 veelken Exp $
//
//

// C++ standard library include files
#include <iostream>
#include <iomanip>
#include <string>

// ROOT include files
#include <TMath.h>

// CMSSW include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

ConeAreaRootFunction::ConeAreaRootFunction()
  : ConeAreaFunction()
{
  coneArea_ = 0;
}

ConeAreaRootFunction::ConeAreaRootFunction(const ConeAreaRootFunction& bluePrint)
  : ConeAreaFunction(bluePrint)
{
  coneArea_ = bluePrint.coneArea_;
}

ConeAreaRootFunction::~ConeAreaRootFunction()
{
//--- nothing to be done yet...
}

//
// assignment operator
//

ConeAreaRootFunction& ConeAreaRootFunction::operator=(const ConeAreaRootFunction& bluePrint)
{
  ConeAreaFunction::operator=(bluePrint);

  coneArea_ = bluePrint.coneArea_;

  return (*this);
}

//
// member functions
//

void ConeAreaRootFunction::SetParameterConeArea(double coneArea)
{
  coneArea_ = coneArea;
}

void ConeAreaRootFunction::SetParameters(double* param)
{
  if ( debugLevel_ > 0 ) {
    edm::LogVerbatim("") << "<ConeAreaRootFunction::SetParameters>:" << std::endl
			 << " theta0 = " << param[0] << std::endl
			 << " phi0 = " << param[1] << std::endl
			 << " coneArea = " << param[2] << std::endl;
  }

  ConeAreaFunction::SetParameters(param);
  
  coneArea_ = param[2];
}

double ConeAreaRootFunction::DoEval(double x) const
{
//--- calculate difference between area covered by cone of opening angle alpha
//    (given as function argument and measured from cone axis)
//    and cone area set as parameter

  return ConeAreaFunction::DoEval(x) - coneArea_;
}  
