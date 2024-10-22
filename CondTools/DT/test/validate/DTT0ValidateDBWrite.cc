#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTT0ValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTT0ValidateHandler> DTT0ValidateDBWrite;

DEFINE_FWK_MODULE(DTT0ValidateDBWrite);
