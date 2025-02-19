#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGBadXTHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGBadXTHandler> ExTestEcalTPGBadXTAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGBadXTAnalyzer);
