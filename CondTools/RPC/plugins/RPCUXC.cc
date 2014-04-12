#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCUXCSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcDataUXC> RPCPopConObUXCAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObUXCAnalyzer);


