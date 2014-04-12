#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCFakeDBGainsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBGainsImpl> CSCFakeGainsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakeGainsPopConAnalyzer);
