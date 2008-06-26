#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "RPCGasSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcGas> RPCGasPAnalyzer;
DEFINE_FWK_MODULE(RPCGasPAnalyzer);


