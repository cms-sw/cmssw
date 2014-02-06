#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTT0Handler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTT0Handler> DTT0PopConAnalyzer;


DEFINE_FWK_MODULE(DTT0PopConAnalyzer);

