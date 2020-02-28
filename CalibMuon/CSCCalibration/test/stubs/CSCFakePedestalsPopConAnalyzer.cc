#include "CSCFakeDBPedestalsHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBPedestalsImpl> CSCFakePedestalsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakePedestalsPopConAnalyzer);
