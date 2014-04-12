#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTPerformanceValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTPerformanceValidateHandler> DTPerformanceValidateDBWrite;


DEFINE_FWK_MODULE(DTPerformanceValidateDBWrite);


