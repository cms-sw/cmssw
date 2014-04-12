#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCDDUMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDDUMapImpl> CSCDDUMapPopConAnalyzer;

DEFINE_FWK_MODULE(CSCDDUMapPopConAnalyzer);
