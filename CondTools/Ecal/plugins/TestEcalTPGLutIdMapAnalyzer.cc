#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGLutIdMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGLutIdMapHandler> ExTestEcalTPGLutIdMapAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGLutIdMapAnalyzer);
