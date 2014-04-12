#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCFebmapSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcDataFebmap> RPCPopConObFebmapAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObFebmapAnalyzer);


