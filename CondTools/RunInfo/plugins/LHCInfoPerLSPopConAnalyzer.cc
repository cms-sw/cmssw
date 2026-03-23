#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LHCInfoPerLSPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<LHCInfoPerLSPopConSourceHandler> LHCInfoPerLSPopConAnalyzer;

DEFINE_FWK_MODULE(LHCInfoPerLSPopConAnalyzer);
