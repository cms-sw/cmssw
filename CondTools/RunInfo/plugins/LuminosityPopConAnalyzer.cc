#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LuminosityInfoHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<lumi::LuminosityInfoHandler> LuminosityPopConAnalyzer;

DEFINE_FWK_MODULE(LuminosityPopConAnalyzer);
