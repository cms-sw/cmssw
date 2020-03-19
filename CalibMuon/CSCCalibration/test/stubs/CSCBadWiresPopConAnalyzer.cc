#include "CSCBadWiresHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCBadWiresImpl> CSCBadWiresPopConAnalyzer;

DEFINE_FWK_MODULE(CSCBadWiresPopConAnalyzer);
