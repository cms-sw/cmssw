#include "CondCore/PopCon/interface/OnlinePopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LHCInfoPerLSPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using LHCInfoPerLSOnlinePopConAnalyzer = popcon::OnlinePopConAnalyzer<LHCInfoPerLSPopConSourceHandler>;

DEFINE_FWK_MODULE(LHCInfoPerLSOnlinePopConAnalyzer);
