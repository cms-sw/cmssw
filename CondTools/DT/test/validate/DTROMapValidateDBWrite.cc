#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTROMapValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTROMapValidateHandler> DTROMapValidateDBWrite;


DEFINE_FWK_MODULE(DTROMapValidateDBWrite);


