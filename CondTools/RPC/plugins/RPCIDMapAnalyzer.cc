#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCIDMapSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::RPCObPVSSmapData> RPCPopConObPVSSmapAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObPVSSmapAnalyzer);
