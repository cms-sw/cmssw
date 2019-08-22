#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCEMapSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::RPCEMapSourceHandler> RPCEMapDBWriter;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCEMapDBWriter);
