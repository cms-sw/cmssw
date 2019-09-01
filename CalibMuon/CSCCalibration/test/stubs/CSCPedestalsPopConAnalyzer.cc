#include "CSCPedestalsHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBPedestalsImpl> CSCPedestalsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCPedestalsPopConAnalyzer);
