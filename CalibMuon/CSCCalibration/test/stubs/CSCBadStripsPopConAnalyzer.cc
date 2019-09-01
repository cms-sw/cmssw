#include "CSCBadStripsHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCBadStripsImpl> CSCBadStripsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCBadStripsPopConAnalyzer);
