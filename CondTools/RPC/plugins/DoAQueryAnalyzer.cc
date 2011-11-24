#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/DoAQuerySH.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::DoAQuerySH> DoAQueryAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(DoAQueryAnalyzer);
