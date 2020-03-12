#include "CSCCrosstalkHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBCrosstalkImpl> CSCCrosstalkPopConAnalyzer;

DEFINE_FWK_MODULE(CSCCrosstalkPopConAnalyzer);
