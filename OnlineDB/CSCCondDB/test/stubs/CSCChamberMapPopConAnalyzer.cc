#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCChamberMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCChamberMapImpl> CSCChamberMapPopConAnalyzer;

DEFINE_FWK_MODULE(CSCChamberMapPopConAnalyzer);
