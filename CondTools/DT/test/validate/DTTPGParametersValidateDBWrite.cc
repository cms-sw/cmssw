#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTTPGParametersValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTTPGParametersValidateHandler> DTTPGParametersValidateDBWrite;


DEFINE_FWK_MODULE(DTTPGParametersValidateDBWrite);


