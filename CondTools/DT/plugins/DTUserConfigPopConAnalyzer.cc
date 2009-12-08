#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTUserConfigHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTUserConfigHandler> DTUserConfigPopConAnalyzer;


DEFINE_FWK_MODULE(DTUserConfigPopConAnalyzer);

