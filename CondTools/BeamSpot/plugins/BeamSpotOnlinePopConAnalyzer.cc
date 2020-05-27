#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/BeamSpot/interface/BeamSpotOnlinePopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<BeamSpotOnlinePopConSourceHandler> BeamSpotOnlinePopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlinePopConAnalyzer);
