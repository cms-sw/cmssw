// Package: CondTools/SiPixel
// Class:   PixelPopConFEDCablingAnalyzer
//
// Description: Analyzer class for pixel fed cabling map
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConFEDCablingAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConFEDCablingSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<PixelPopConFEDCablingSourceHandler> PixelPopConFEDCablingAnalyzer;

DEFINE_FWK_MODULE(PixelPopConFEDCablingAnalyzer);
