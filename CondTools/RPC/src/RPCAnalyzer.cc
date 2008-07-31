#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcData> RPCPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConAnalyzer);


