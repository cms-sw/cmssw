#ifndef PhysicsTools_IsolationUtils_FixedAreaIsolationCone_h
#define PhysicsTools_IsolationUtils_FixedAreaIsolationCone_h

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
// $Id: FixedAreaIsolationCone.h,v 1.2 2009/01/14 10:53:14 hegner Exp $
//
//

// ROOT include files
#include <Math/RootFinder.h>
#include <Math/RootFinderAlgorithms.h>

// CMSSW include files
#include "PhysicsTools/IsolationUtils/interface/ConeAreaFunction.h"
#include "PhysicsTools/IsolationUtils/interface/ConeAreaRootFunction.h"

//
// class declaration
//

class FixedAreaIsolationCone
{
 public:
  // default constructor
  FixedAreaIsolationCone();
  
  // destructor
  ~FixedAreaIsolationCone();

  // set acceptance limit for particle reconstruction
  // (this will enlarge the isolation cone near the acceptance boundary
  //  such that the area in the region where particles can be reconstructed is constant;
  //  i.e. guarantees flat efficiency near the acceptance boundary)
  void setAcceptanceLimit(double etaMaxTrackingAcceptance);
  
  double operator() (double coneAxisTheta, double coneAxisPhi,
		     double openingAngleSignalCone, double areaIsolationCone, int& error);
  
 private:
  ConeAreaFunction areaFunctionSignalCone_; 
  ConeAreaRootFunction areaRootFunctionIsolationCone_;
  ROOT::Math::RootFinder areaRootFinderIsolationCone_; 

  static const unsigned int debugLevel_ = 0;
};

#endif
