#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCVmonSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcDataV> RPCPopConObCondAnalyzerV;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObCondAnalyzerV);


