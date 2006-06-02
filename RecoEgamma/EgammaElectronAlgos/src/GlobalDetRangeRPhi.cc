// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      GlobalDetRangeRPhi
// 
/**\class ElectronPixelSeedProducer EgammaElectronAlgos/GlobalDetRangeRPhi

 Description: MeasurementEstimator for Pixel Barrel, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id$
//
//
using namespace std;  //FIXME, for PhiLess
#include "RecoEgamma/EgammaElectronAlgos/src/GlobalDetRangeRPhi.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Surface/interface/Bounds.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include <vector>


GlobalDetRangeRPhi::GlobalDetRangeRPhi( const BoundPlane& plane) {

  float dx = plane.bounds().width()/2.;
  float dy = plane.bounds().length()/2.;

  vector<Surface::GlobalPoint> corners(4);
  corners[0] = plane.toGlobal( LocalPoint( -dx, -dy, 0));
  corners[1] = plane.toGlobal( LocalPoint( -dx,  dy, 0));
  corners[2] = plane.toGlobal( LocalPoint(  dx, -dy, 0));
  corners[3] = plane.toGlobal( LocalPoint(  dx,  dy, 0));

  float phimin = corners[0].phi();  float phimax = phimin;
  float rmin   = corners[0].perp(); float rmax   = rmin;
  for ( int i=1; i<4; i++) {
    float phi = corners[i].phi();
    if ( PhiLess()( phi, phimin)) phimin = phi;
    if ( PhiLess()( phimax, phi)) phimax = phi;

//    if ( phi < phimin) phimin = phi;
//    if ( phi > phimax) phimax = phi;

    float r = corners[i].perp();
    if ( r < rmin) rmin = r;
    if ( r > rmax) rmax = r;
  }

  theRRange.first    = rmin;
  theRRange.second   = rmax;
  thePhiRange.first  = phimin;
  thePhiRange.second = phimax;

}
