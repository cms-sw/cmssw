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
// $Id: FTSFromVertexToPointFactory.cc,v 1.6 2011/03/21 17:10:32 innocent Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "MagneticField/Engine/interface/MagneticField.h"


FreeTrajectoryState FTSFromVertexToPointFactory::operator()(const MagneticField *magField, const GlobalPoint& xmeas,  
                                                            const GlobalPoint& xvert, 
                                                            float momentum, 
							    TrackCharge charge)
{
  double BInTesla = magField->inTesla(xmeas).z();
  GlobalVector xdiff = xmeas -xvert;
  double theta = xdiff.theta();
  double phi= xdiff.phi();
  double pt = momentum*sin(theta);
  double pz = momentum*cos(theta);
  double pxOld = pt*cos(phi);
  double pyOld = pt*sin(phi);

  double RadCurv = 100*pt/(BInTesla*0.29979);
  double alpha = asin(0.5*xdiff.perp()/RadCurv);

  float ca = cos(charge*alpha);
  float sa = sin(charge*alpha);
  double pxNew =   ca*pxOld + sa*pyOld;
  double pyNew =  -sa*pxOld + ca*pyOld;
  GlobalVector pNew(pxNew, pyNew, pz);  

  GlobalTrajectoryParameters gp(xmeas, pNew, charge, magField);
  
  AlgebraicSymMatrix55 C = AlgebraicMatrixID();
  FreeTrajectoryState VertexToPoint(gp,CurvilinearTrajectoryError(C));

  return VertexToPoint;
}






