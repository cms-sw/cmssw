#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/L1RPCHwConfigSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::L1RPCHwConfigSourceHandler> L1RPCHwConfigDBWriter;
//define this as a plug-in
DEFINE_FWK_MODULE(L1RPCHwConfigDBWriter);
