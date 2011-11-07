#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCNoiseSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::RPCNoiseSH> RPCNoiseAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCNoiseAnalyzer);
