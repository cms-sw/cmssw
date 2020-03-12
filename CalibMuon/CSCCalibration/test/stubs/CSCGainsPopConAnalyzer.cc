#include "CSCGainsHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBGainsImpl> CSCGainsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCGainsPopConAnalyzer);
