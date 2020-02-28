#include "CSCL1TPParametersHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCL1TPParametersImpl> CSCL1TPParametersPopConAnalyzer;

DEFINE_FWK_MODULE(CSCL1TPParametersPopConAnalyzer);
