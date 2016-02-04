#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCGainsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBGainsImpl> CSCGainsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCGainsPopConAnalyzer);
