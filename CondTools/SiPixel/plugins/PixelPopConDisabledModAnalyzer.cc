// Package: CondTools/SiPixel
// Class:   PixelPopConDisableModAnalyzer
//
// Description: Analyzer class for pixel disabled module
//              popcon application
//
// Created: M. Eads, Apr 2008

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConDisabledModAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConDisabledModSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<PixelPopConDisabledModSourceHandler> PixelPopConDisabledModAnalyzer;

DEFINE_FWK_MODULE(PixelPopConDisabledModAnalyzer);
