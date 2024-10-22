#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTHVStatusValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTHVStatusValidateHandler> DTHVStatusValidateDBWrite;

DEFINE_FWK_MODULE(DTHVStatusValidateDBWrite);
