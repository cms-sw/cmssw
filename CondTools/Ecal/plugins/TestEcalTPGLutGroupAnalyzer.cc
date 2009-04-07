#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGLutGroupHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGLutGroupHandler> ExTestEcalTPGLutGroupAnalyzer;



//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGLutGroupAnalyzer);
