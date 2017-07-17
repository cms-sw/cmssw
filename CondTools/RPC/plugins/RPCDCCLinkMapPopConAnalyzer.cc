#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCDCCLinkMapHandler.h"

typedef popcon::PopConAnalyzer<RPCDCCLinkMapHandler> RPCDCCLinkMapPopConAnalyzer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCDCCLinkMapPopConAnalyzer);
