// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelMatchStartLayers
// 
/**\class PixelMatchStartLayers EgammaElectronAlgos/PixelMatchStartLayers

 Description: class to find the innermost pixel forward layers

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelMatchStartLayers.cc,v 1.1 2006/06/02 16:21:02 uberthon Exp $
//
//

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include <iostream> 
#include <algorithm>
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchStartLayers.h"

PixelMatchStartLayers::PixelMatchStartLayers() {}

void PixelMatchStartLayers::setup(const GeometricSearchTracker * layers) {
  barrelPixel = layers->pixelBarrelLayers();
  posPixel = layers->posPixelForwardLayers();
  negPixel = layers->negPixelForwardLayers(); 
}













