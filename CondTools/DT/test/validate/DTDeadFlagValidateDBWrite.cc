#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTDeadFlagValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTDeadFlagValidateHandler> DTDeadFlagValidateDBWrite;


DEFINE_FWK_MODULE(DTDeadFlagValidateDBWrite);


