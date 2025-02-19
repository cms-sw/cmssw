#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCChamberIndexHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

typedef popcon::PopConAnalyzer<popcon::CSCChamberIndexImpl> CSCChamberIndexPopConAnalyzer;

DEFINE_FWK_MODULE(CSCChamberIndexPopConAnalyzer);
