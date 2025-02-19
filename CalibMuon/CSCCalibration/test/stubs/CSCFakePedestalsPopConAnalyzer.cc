#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCFakeDBPedestalsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCFakeDBPedestalsImpl> CSCFakePedestalsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCFakePedestalsPopConAnalyzer);
