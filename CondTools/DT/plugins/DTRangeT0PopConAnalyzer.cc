#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTRangeT0Handler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTRangeT0Handler> DTRangeT0PopConAnalyzer;


DEFINE_FWK_MODULE(DTRangeT0PopConAnalyzer);

