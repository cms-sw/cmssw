// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      FTSFromVertexToPointFactory
// 
/**\class FTSFromVertexToPointFactory EgammaElectronAlgos/FTSFromVertexToPointFactory

 Description: Utility class to create FTS from supercluster

 Implementation:
     should go somewhere else in the future
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "MagneticField/Engine/interface/MagneticField.h"


FreeTrajectoryState FTSFromVertexToPointFactory::get( MagneticField const & magField, 
                                                      GlobalPoint const & xmeas, 
                                                      GlobalPoint const & xvert, 
                                                      float momentum, 
                                                      TrackCharge charge )
{
  auto BInTesla = magField.inTesla(xmeas).z();
  GlobalVector xdiff = xmeas - xvert;
  auto mom = momentum*xdiff.unit();
  auto pt = mom.perp();
  auto pz = mom.z();
  auto pxOld = mom.x();
  auto pyOld = mom.y();


  auto curv = (BInTesla*0.29979f*0.01f)/pt;

  // stays as doc...
  // auto alpha = std::asin(0.5f*xdiff.perp()*curv);
  // auto ca = std::cos(float(charge)*alpha);
  // auto sa = std::sin(float(charge)*alpha);
 
  auto sa = 0.5f*xdiff.perp()*curv*float(charge);
  auto ca = sqrt(1.f-sa*sa); 

  auto pxNew =   ca*pxOld + sa*pyOld;
  auto pyNew =  -sa*pxOld + ca*pyOld;
  GlobalVector pNew(pxNew, pyNew, pz);  

  GlobalTrajectoryParameters gp(xmeas, pNew, charge, & magField);
  
  return FreeTrajectoryState(gp);
}
