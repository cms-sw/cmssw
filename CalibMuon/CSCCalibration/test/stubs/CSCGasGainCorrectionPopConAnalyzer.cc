#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCDBGasGainCorrectionHandler.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBGasGainCorrectionImpl> CSCDBGasGainCorrectionPopConAnalyzer;

DEFINE_FWK_MODULE(CSCDBGasGainCorrectionPopConAnalyzer);
