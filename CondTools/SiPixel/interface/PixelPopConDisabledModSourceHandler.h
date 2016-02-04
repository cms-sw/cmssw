#ifndef PIXELPOPCONDISABLEDMODSOURCEHANDLER_H
#define PIXELPOPCONDISABLEDMODSOURCEHANDLER_H

// Package: CondTools/SiPixel
// Class:   PixelPopConDisableModSourceHandler
//
// Description: Source handler class for pixel disabled module
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondTools/SiPixel/interface/PixelPopConSourceHandler.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDisabledModules.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class definition
class PixelPopConDisabledModSourceHandler : public PixelPopConSourceHandler<SiPixelDisabledModules> {

 public:
  PixelPopConDisabledModSourceHandler(edm::ParameterSet const &) {;}

 private:

};


#endif
