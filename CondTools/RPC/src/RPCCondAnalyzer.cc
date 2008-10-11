#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCCondSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcData> RPCPopConObCondAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObCondAnalyzer);


