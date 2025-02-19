#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCFakeDBNoiseMatrixHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBNoiseMatrixImpl> CSCFakeNoiseMatrixPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakeNoiseMatrixPopConAnalyzer);
