#include "CondCore/PopCon/interface/OnlinePopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LHCInfoPerFillPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::OnlinePopConAnalyzer<LHCInfoPerFillPopConSourceHandler> LHCInfoPerFillOnlinePopConAnalyzer;

DEFINE_FWK_MODULE(LHCInfoPerFillOnlinePopConAnalyzer);
