#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGBadTTHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGBadTTHandler> ExTestEcalTPGBadTTAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGBadTTAnalyzer);
