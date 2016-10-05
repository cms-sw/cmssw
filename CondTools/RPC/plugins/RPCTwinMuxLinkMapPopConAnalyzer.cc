#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCTwinMuxLinkMapHandler.h"

typedef popcon::PopConAnalyzer<RPCTwinMuxLinkMapHandler> RPCTwinMuxLinkMapPopConAnalyzer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCTwinMuxLinkMapPopConAnalyzer);
