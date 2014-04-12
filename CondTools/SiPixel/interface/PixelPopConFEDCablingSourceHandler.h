#ifndef PIXELPOPCONFEDCABLINGSOURCEHANDLER_H
#define PIXELPOPCONFEDCABLINGSOURCEHANDLER_H

// Package: CondTools/SiPixel
// Class:   PixelPopConFEDCablingSourceHandler
//
// Description: Source handler class for pixel fed cabling map
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondTools/SiPixel/interface/PixelPopConSourceHandler.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class definition
class PixelPopConFEDCablingSourceHandler : public PixelPopConSourceHandler<SiPixelFedCablingMap> {

 public:
  PixelPopConFEDCablingSourceHandler(edm::ParameterSet const &) {;}

 private:

};


#endif
