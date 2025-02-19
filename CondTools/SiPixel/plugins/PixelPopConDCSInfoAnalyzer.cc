// Package: CondTools/SiPixel
// Class:   PixelPopConDCSInfoAnalyzer
//
// Description: Analyzer class for pixel dcs info
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConDCSInfoAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConDCSInfoSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<PixelPopConDCSInfoSourceHandler> PixelPopConDCSInfoAnalyzer;

DEFINE_FWK_MODULE(PixelPopConDCSInfoAnalyzer);
