#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTPerformanceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTPerformanceHandler> DTPerformancePopConAnalyzer;


DEFINE_FWK_MODULE(DTPerformancePopConAnalyzer);

