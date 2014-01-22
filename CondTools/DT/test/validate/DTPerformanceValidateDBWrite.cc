#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTPerformanceValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTPerformanceValidateHandler> DTPerformanceValidateDBWrite;


DEFINE_FWK_MODULE(DTPerformanceValidateDBWrite);


