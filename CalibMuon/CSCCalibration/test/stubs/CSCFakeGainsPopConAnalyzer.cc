#include "CSCFakeDBGainsHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBGainsImpl> CSCFakeGainsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakeGainsPopConAnalyzer);
