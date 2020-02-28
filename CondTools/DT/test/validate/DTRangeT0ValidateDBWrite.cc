#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTRangeT0ValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTRangeT0ValidateHandler> DTRangeT0ValidateDBWrite;

DEFINE_FWK_MODULE(DTRangeT0ValidateDBWrite);
