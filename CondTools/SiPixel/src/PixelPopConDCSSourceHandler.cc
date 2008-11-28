#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondTools/SiPixel/interface/PixelPopConDCSSourceHandler.h"

typedef popcon::PopConAnalyzer< PixelPopConDCSSourceHandler<bool> > PixelPopConBoolAnalyzer;
typedef popcon::PopConAnalyzer< PixelPopConDCSSourceHandler<float> > PixelPopConFloatAnalyzer;

DEFINE_FWK_MODULE(PixelPopConBoolAnalyzer);
DEFINE_FWK_MODULE(PixelPopConFloatAnalyzer);
