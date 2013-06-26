#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTTtrigValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTTtrigValidateHandler> DTTtrigValidateDBWrite;


DEFINE_FWK_MODULE(DTTtrigValidateDBWrite);


