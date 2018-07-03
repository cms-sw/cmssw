#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCLBLinkMapHandler.h"

typedef popcon::PopConAnalyzer<RPCLBLinkMapHandler> RPCLBLinkMapPopConAnalyzer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCLBLinkMapPopConAnalyzer);
