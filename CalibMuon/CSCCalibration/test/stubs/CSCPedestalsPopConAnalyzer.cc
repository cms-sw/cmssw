#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCPedestalsHandler.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBPedestalsImpl> CSCPedestalsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCPedestalsPopConAnalyzer);
