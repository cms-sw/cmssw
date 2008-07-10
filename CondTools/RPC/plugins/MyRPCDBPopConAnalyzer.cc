#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCDBPerformanceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<RPCDBPerformanceHandler> MyRPCDBPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(MyRPCDBPopConAnalyzer);
