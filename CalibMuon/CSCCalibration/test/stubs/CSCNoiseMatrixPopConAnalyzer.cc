#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCNoiseMatrixHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBNoiseMatrixImpl> CSCNoiseMatrixPopConAnalyzer;

DEFINE_FWK_MODULE(CSCNoiseMatrixPopConAnalyzer);
