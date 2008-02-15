#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTCCBConfigHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTCCBConfigHandler> DTCCBConfigPopConAnalyzer;


DEFINE_FWK_MODULE(DTCCBConfigPopConAnalyzer);

