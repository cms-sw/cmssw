#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LHCInfoPerLSPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<LHCInfoPerLSPopConSourceHandler> LHCInfoPerLSPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(LHCInfoPerLSPopConAnalyzer);
