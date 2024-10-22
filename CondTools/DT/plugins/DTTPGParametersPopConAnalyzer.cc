#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTTPGParametersHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTTPGParametersHandler> DTTPGParametersPopConAnalyzer;

DEFINE_FWK_MODULE(DTTPGParametersPopConAnalyzer);
