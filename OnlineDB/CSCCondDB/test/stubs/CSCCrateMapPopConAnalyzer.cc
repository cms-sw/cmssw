#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCCrateMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCCrateMapImpl> CSCCrateMapPopConAnalyzer;

DEFINE_FWK_MODULE(CSCCrateMapPopConAnalyzer);


