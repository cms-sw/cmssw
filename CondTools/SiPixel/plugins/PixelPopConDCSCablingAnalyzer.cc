// Package: CondTools/SiPixel
// Class:   PixelPopConDCSCablingAnalyzer
//
// Description: Analyzer class for pixel dcs cabling map
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConDCSCablingAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConDCSCablingSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<PixelPopConDCSCablingSourceHandler> PixelPopConDCSCablingAnalyzer;

DEFINE_FWK_MODULE(PixelPopConDCSCablingAnalyzer);
