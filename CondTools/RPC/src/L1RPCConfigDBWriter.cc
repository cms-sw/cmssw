#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/L1RPCConfigSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::L1RPCConfigSourceHandler> L1RPCConfigDBWriter;
//      void analyze(const edm::Event& evt, const edm::EventSetup& est);
//define this as a plug-in
DEFINE_FWK_MODULE(L1RPCConfigDBWriter);

