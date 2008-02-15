#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTMtimeHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTMtimeHandler> DTMtimePopConAnalyzer;


DEFINE_FWK_MODULE(DTMtimePopConAnalyzer);

