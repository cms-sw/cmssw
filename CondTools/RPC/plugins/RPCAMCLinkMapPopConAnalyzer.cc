#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCAMCLinkMapHandler.h"

typedef popcon::PopConAnalyzer<RPCAMCLinkMapHandler> RPCAMCLinkMapPopConAnalyzer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCAMCLinkMapPopConAnalyzer);
