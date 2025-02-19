#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCFakeDBCrosstalkHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBCrosstalkImpl> CSCFakeCrosstalkPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakeCrosstalkPopConAnalyzer);
