#ifndef PIXELPOPCONDCSINFOSOURCEHANDLER_H
#define PIXELPOPCONDCSINFOSOURCEHANDLER_H

// Package: CondTools/SiPixel
// Class:   PixelPopConDCSInfoSourceHandler
//
// Description: Source handler class for pixel dcs info
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondTools/SiPixel/interface/PixelPopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class definition
// no object for dcs cabling map, yet, so just stick in a bool
class PixelPopConDCSInfoSourceHandler : public PixelPopConSourceHandler<bool> {

 public:
  PixelPopConDCSInfoSourceHandler(edm::ParameterSet const &) {;}

 private:

};


#endif
