#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCChamberTimeCorrectionsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCChamberTimeCorrectionsImpl> CSCChamberTimeCorrectionsPopConAnalyzer;

DEFINE_FWK_MODULE(CSCChamberTimeCorrectionsPopConAnalyzer);
