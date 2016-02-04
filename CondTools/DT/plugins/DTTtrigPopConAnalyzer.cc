#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTTtrigHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTTtrigHandler> DTTtrigPopConAnalyzer;


DEFINE_FWK_MODULE(DTTtrigPopConAnalyzer);

