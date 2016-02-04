#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCDBL1TPParametersHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBL1TPParametersImpl> CSCDBL1TPParametersPopConAnalyzer;

DEFINE_FWK_MODULE(CSCDBL1TPParametersPopConAnalyzer);
