#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Luminosity/interface/LumiSectionDataHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<lumi::LumiSectionDataHandler> LumiSectionDataPopCon;

DEFINE_FWK_MODULE(LumiSectionDataPopCon);
