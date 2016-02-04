#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCBadChambersHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCBadChambersImpl> CSCBadChambersPopConAnalyzer;

DEFINE_FWK_MODULE(CSCBadChambersPopConAnalyzer);
