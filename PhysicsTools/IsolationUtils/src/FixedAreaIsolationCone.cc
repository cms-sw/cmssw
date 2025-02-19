#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"

// -*- C++ -*-
//
// Package:    
// Class:      FixedAreaIsolationCone
// 
/**\class FixedAreaIsolationCone FixedAreaIsolationCone.cc PhysicsTools/IsolationUtils/src/FixedAreaIsolationCone.cc

 Description: highest level class to compute size of isolation cone 
              such that area weighted by particle density 
              (proportional to dEta/dTheta = 1/sin(theta)) 
              is constant

 Implementation:
     imported into CMSSW on 05/18/2007
*/
//
// Original Author: Christian Veelken, UC Davis
//         Created: Wed May 16 13:47:40 CST 2007
// $Id: FixedAreaIsolationCone.cc,v 1.1 2007/05/23 20:21:38 veelken Exp $
//
//

// C++ standard library include files
#include <string>

// ROOT include files
#include <TMath.h>

// CMSSW include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

FixedAreaIsolationCone::FixedAreaIsolationCone()
  : areaFunctionSignalCone_(), areaRootFunctionIsolationCone_()
{
//--- nothing to be done yet
//
//    WARNING: do NOT call ROOT::Math::RootFinder<ROOT::Math::Roots::Brent>::SetFunction here;
//             this will cause the function to be evaluated before all function parameters are set,
//             leading to an error message first and erroneous behaviour of the root-finding later on !!!
//
}

FixedAreaIsolationCone::~FixedAreaIsolationCone()
{
//--- nothing to be done yet
}

void FixedAreaIsolationCone::setAcceptanceLimit(double etaMaxTrackingAcceptance)
{
  areaFunctionSignalCone_.SetAcceptanceLimit(etaMaxTrackingAcceptance);
  areaRootFunctionIsolationCone_.SetAcceptanceLimit(etaMaxTrackingAcceptance);
}

double FixedAreaIsolationCone::operator() (double coneAxisTheta, double coneAxisPhi,
					   double openingAngleSignalCone, double areaIsolationCone, int& error)
{
  areaFunctionSignalCone_.SetParameterTheta0(coneAxisTheta);
  areaFunctionSignalCone_.SetParameterPhi0(coneAxisPhi);
  double areaSignalCone = areaFunctionSignalCone_(openingAngleSignalCone);

  areaRootFunctionIsolationCone_.SetParameterTheta0(coneAxisTheta);
  areaRootFunctionIsolationCone_.SetParameterPhi0(coneAxisPhi);
  areaRootFunctionIsolationCone_.SetParameterConeArea(areaIsolationCone + areaSignalCone);
  areaRootFinderIsolationCone_.SetFunction(areaRootFunctionIsolationCone_, 0. , TMath::Pi());
  int statusIsolationCone = areaRootFinderIsolationCone_.Solve();
  double openingAngleIsolationCone = areaRootFinderIsolationCone_.Root();

  if ( debugLevel_ > 0 ) {
    const std::string category = "FixedAreaIsolationCone::operator()";
    edm::LogVerbatim(category) << "openingAngleSignalCone = " << openingAngleSignalCone << std::endl;
    edm::LogVerbatim(category) << "areaSignalCone = " << areaSignalCone << std::endl;
    edm::LogVerbatim(category) << "areaIsolationCone = " << areaIsolationCone << std::endl;
    edm::LogVerbatim(category) << "openingAngleIsolationCone = " << openingAngleIsolationCone << std::endl;
    edm::LogVerbatim(category) << "statusIsolationCone = " << statusIsolationCone << std::endl;
  }

  if ( statusIsolationCone == 0 ) { 
    error = 0;
    return openingAngleIsolationCone;
  } else {
    error = 1;
    return 0.;
  }
}
