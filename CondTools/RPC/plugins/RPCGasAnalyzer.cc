#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCGasSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcObGasData> RPCPopConObGasAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObGasAnalyzer);


