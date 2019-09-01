#include "CSCFakeDBNoiseMatrixHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBNoiseMatrixImpl> CSCFakeNoiseMatrixPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakeNoiseMatrixPopConAnalyzer);
