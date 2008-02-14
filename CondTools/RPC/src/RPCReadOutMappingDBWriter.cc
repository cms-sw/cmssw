#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCReadOutMappingSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::RPCReadOutMappingSourceHandler> RPCReadOutMappingDBWriter;
//define this as a plug-in
DEFINE_FWK_MODULE(RPCReadOutMappingDBWriter);

