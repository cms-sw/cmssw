#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/test/validate/DTCCBConfigValidateHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTCCBConfigValidateHandler> DTCCBConfigValidateDBWrite;


DEFINE_FWK_MODULE(DTCCBConfigValidateDBWrite);


