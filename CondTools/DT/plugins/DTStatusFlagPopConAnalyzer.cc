#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTStatusFlagHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTStatusFlagHandler> DTStatusFlagPopConAnalyzer;


DEFINE_FWK_MODULE(DTStatusFlagPopConAnalyzer);

