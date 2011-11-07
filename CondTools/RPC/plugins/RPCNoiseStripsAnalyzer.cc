#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCNoiseStripsSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::RPCNoiseStripsSH> RPCNoiseStripsAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCNoiseStripsAnalyzer);
