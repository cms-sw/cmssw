#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCGasMixSH.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::RpcDataGasMix> RPCPopConObGasMixAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(RPCPopConObGasMixAnalyzer);


