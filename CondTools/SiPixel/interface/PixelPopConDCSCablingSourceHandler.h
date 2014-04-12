#ifndef PIXELPOPCONDCSCABLINGSOURCEHANDLER_H
#define PIXELPOPCONDCSCABLINGSOURCEHANDLER_H

// Package: CondTools/SiPixel
// Class:   PixelPopConDCSCablingSourceHandler
//
// Description: Source handler class for pixel dcs cabling map
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondTools/SiPixel/interface/PixelPopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class definition
// no object for dcs cabling map, yet, so just stick in a bool
class PixelPopConDCSCablingSourceHandler : public PixelPopConSourceHandler<bool> {

 public:
  PixelPopConDCSCablingSourceHandler(edm::ParameterSet const &) {;}

 private:

};


#endif
