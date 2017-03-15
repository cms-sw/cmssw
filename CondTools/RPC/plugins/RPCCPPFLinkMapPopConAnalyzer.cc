#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCCPPFLinkMapHandler.h"

typedef popcon::PopConAnalyzer<RPCCPPFLinkMapHandler> RPCCPPFLinkMapPopConAnalyzer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCCPPFLinkMapPopConAnalyzer);
