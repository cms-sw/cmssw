#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCTempSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcDataT> RPCPopConObCondAnalyzerT;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObCondAnalyzerT);


