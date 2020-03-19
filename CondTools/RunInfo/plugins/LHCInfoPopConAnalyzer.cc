#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LHCInfoPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<LHCInfoPopConSourceHandler> LHCInfoPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(LHCInfoPopConAnalyzer);
