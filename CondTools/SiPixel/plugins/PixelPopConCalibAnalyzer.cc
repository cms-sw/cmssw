
// -*- C++ -*-
//
// Package:    CondTools/SiPixel
// Class:      PixelPopConCalibAnalyzer
// 
/**\class PixelPopConCalibAnalyzer PixelPopConCalibAnalyzer.cc CondTools/SiPixel/src/PixelPopConCalibAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Eads
//         Created:  8 Feb 2008
// $Id: PixelPopConCalibAnalyzer.cc,v 1.1 2009/05/28 22:14:22 dlange Exp $
//
//


#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiPixel/interface/PixelPopConCalibAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/SiPixel/interface/PixelPopConCalibSourceHandler.h"


typedef popcon::PopConAnalyzer<PixelPopConCalibSourceHandler> PixelPopConCalibAnalyzer;

DEFINE_FWK_MODULE(PixelPopConCalibAnalyzer);
