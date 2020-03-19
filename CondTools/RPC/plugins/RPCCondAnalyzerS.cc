#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCStatusSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::RpcDataS> RPCPopConObCondAnalyzerS;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObCondAnalyzerS);
