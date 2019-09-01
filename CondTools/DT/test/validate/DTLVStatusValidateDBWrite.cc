#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTLVStatusValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTLVStatusValidateHandler> DTLVStatusValidateDBWrite;

DEFINE_FWK_MODULE(DTLVStatusValidateDBWrite);
