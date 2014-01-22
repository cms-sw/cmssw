#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTMtimeValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTMtimeValidateHandler> DTMtimeValidateDBWrite;


DEFINE_FWK_MODULE(DTMtimeValidateDBWrite);


