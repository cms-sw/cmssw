#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTHVStatusHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTHVStatusHandler> DTHVStatusPopConAnalyzer;


DEFINE_FWK_MODULE(DTHVStatusPopConAnalyzer);

