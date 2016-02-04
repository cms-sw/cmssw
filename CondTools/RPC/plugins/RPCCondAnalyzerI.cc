#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCImonSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcDataI> RPCPopConObCondAnalyzerI;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObCondAnalyzerI);


