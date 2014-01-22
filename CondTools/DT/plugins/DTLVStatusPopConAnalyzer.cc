#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTLVStatusHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTLVStatusHandler> DTLVStatusPopConAnalyzer;


DEFINE_FWK_MODULE(DTLVStatusPopConAnalyzer);

