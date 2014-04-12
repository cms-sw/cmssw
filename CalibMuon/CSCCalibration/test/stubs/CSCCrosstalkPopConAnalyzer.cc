#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCCrosstalkHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBCrosstalkImpl> CSCCrosstalkPopConAnalyzer;

DEFINE_FWK_MODULE(CSCCrosstalkPopConAnalyzer);
