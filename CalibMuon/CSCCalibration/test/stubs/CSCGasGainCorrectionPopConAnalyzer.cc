#include "CSCDBGasGainCorrectionHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBGasGainCorrectionImpl> CSCDBGasGainCorrectionPopConAnalyzer;

DEFINE_FWK_MODULE(CSCDBGasGainCorrectionPopConAnalyzer);
